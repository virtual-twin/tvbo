# TODO

## Migrate runtime ontology: deprecated `tvb-o.owl` → generated `tvbo.owl`

Switch every runtime consumer from the deprecated **class-based**
`tvbo/data/ontology/tvb-o.owl` (1516 classes / 173 individuals; `JansenRit`
is an `owl:Class`) to the generated **individual-based** `ontology/tvbo.owl`
(422 classes / 1236 individuals; `JansenRit` is an `owl:NamedIndividual` with
explicit `tvbo:hasParameter`/`hasDerivedVariable` edges), built by
`make gen-merged`. **Preserve** the deprecated file — do not overwrite it; keep
it as a frozen parity reference.

**Phase A (platform) — ✅ DONE.** The platform's `DirectOntologyAPI` now serves
the generated ontology from a *dedicated `owlready2.World`* (a global loader
repoint crashes the tvbo core — base-IRI mismatch + empty class queries — so
`owl.py`'s global `onto` stays on the deprecated ontology). `make gen-merged`
packages `tvbo/data/ontology/tvbo.owl`; `query.py` gained optional `onto=`/`world=`
args; enrichment/search/hierarchy verified; tvbo core untouched.

**Phase B (TODO) — with `owl.py`. Bigger than first scoped.** A 2026-07-01 code
audit found the generated ontology diverges from the deprecated one far beyond
"classes → individuals": **no `NeuralMassModel` class** (models are `Dynamics`
individuals; the `functional_models` allow-list drives the 22 — NOT
`model_type=='neural_mass'`, which yields only 14); scaffold vocabulary
renamed/deleted (`IntegrationMethod`→`Integrator`, `Constant`/`TimeDerivative`/
`CouplingTerm`/… gone → import-time crashes); the has* traversal edges are
**AnnotationProperties** (SPARQL-as-relation fails); properties missing/renamed
(`symbol`→`skos:notation`, equation→`tvbo:rhs`/`lhs`, `synonym`→`skos:altLabel`,
no `range`/`VOIs`/`has_cvar`); **verbose labels** (key on `skos:notation`;
`_<ACR>` suffix scheme dead); derivatives inline on StateVariable; references
→`dcterms:references`/`studies/`; ~40 punned props silently coerced. Scope: **43
class-based owl.py functions + the writer (`import_model`) + ~20 downstream
consumers (main codegen path `codegen/templater.py`) + 4 import-time
statements**, then retire the isolated world (inverting the Phase-A
`api.world is not owl.onto.world` invariant). Full file-by-file spec, impact map,
and verification: **see `dev/runtime_ontology_migration.md` §3.0–§5**.

**Design principle: three explicit load routes; YAML supervenes (from PR #43).**
A model / experiment / class spec can be obtained three ways, all supported:
1. **From ontology** — explicit (`Dynamics.from_ontology`).
2. **From YAML / string / metadata** — the default
   (`from_file`/`from_string`/`from_datamodel`/`from_db`, `use_ontology=False`).
3. **From YAML enriched by ontology** — explicit opt-in (`use_ontology=True` /
   `enrich_from_ontology()`).

Rules: **YAML supervenes** — by default the ontology is not touched; **enrichment
is NOT the default** and, when requested, only **fills missing pieces** (never
overrides a value present in the YAML). The load side already honours this
(default `use_ontology=False`); TODO: verify `enrich_from_ontology` /
`_populate_from_ontology` are strictly gap-fill and never clobber YAML.

Concrete route-3 example — a component referenced by `iri` draws its spec from
the ontology, with inline metadata overriding on top (YAML supervenes):

    SimulationExperiment = {
      network: { dynamics: { g2d: {
        iri: "tvbo:Generic2dOscillator",
        parameters: { a: { value: 1 } },
      }}}
    }

`iri` pulls Generic2dOscillator's equations + full parameter set + defaults from
the ontology; the inline `a: {value: 1}` overrides *only* the default for `a`,
while every other parameter and the equations are fetched from the ontology. The
same applies to any IRI-sourced component (coupling, …). Enrichment here is
still opt-in (triggered by the presence of `iri`), and inline values always win.

**Generalize to every concept/entity.** These three routes are not Dynamics-
specific — they should be the *uniform* resolution contract for every component a
`SimulationExperiment` references (coupling, network / atlas / parcellation /
tractogram, integration, observation, study, …). Pieces already exist
generically: `registry.resolve(cls, name)`, CURIE stripping (`registry.local_name`),
and the `iri`→registry-YAML backfill in `experiment.py`; the `iri` prefix already
selects the source (`tvbo:` → ontology/DB, `neuroml:` → NeuroML, …). But today the
routes are re-implemented per class (`from_db`/`from_ontology`/`use_ontology` on
coupling, dynamics, network, observation, noise, study, perturbation, continuation,
experiment) and enrichment is coarse — collection-level fill-if-absent (e.g.
`experiment.py` injects `parameters` only when none are given), not field-level.
Standardize to a single generic resolver every component passes through, doing a
**recursive (leaf-level) merge** so inline values win at the field (the
`a: {value: 1}` example overrides only `a`, keeping every other ontology-sourced
parameter and the equations). One code path replaces the per-class duplication.

The remaining violation is **codegen**. The live path is the export registry →
`Dynamics.render_code` rendering `tvbo-tvb-model.py.mako` (dynamics.py:2043),
which is passed the YAML `Dynamics` but still derives coupling names via
`_onto.get_model_coupling_terms(model.name)` (route 1). Route-2 fix (localized to
that template): derive global coupling inputs from `model.coupling_inputs`
(filter `local`/`local_coupling`) instead — then TVB emits `c_glob` consistently
instead of mixing YAML equations (`c_glob`) with stale ontology coupling terms
(`c_pop0` → empty `coupling_terms`, undefined coupling symbol).

**Retire dead codegen:** `tvbo/codegen/templater.py`'s model-render path is dead
(0 callers): `model2class`, `get_model_info`, `get_statevariable_equations`,
`get_param_info`, `get_sv_info`, `equation2class`, `coupling2class`,
`integrator2class`, plus `tvbo/templates/_tvbo-tvb-model_old.py.mako`. Remove
these; keep/relocate the still-used helpers `format_code`, `exec_globals`,
`source_observations`, `is_derived`, `get_integrator_info`.

PR #43 added a stopgap, `owl._sync_model_from_yaml` (rebuild a model's ontology
classes from YAML on every `get_model`). The review confirmed it is the wrong
layer and buggy — do NOT keep it; Phase B replaces it:
- `import_model` names coupling-term classes without a model suffix, so a bare
  `c_glob`/`c_glob0` is a single shared owlready2 object across models; the
  runtime re-import re-parents/destroys it and silently strips an earlier
  model's global coupling (`get_model_coupling_terms('JansenRit')` →
  `['local_coupling']` after resolving `Generic2dOscillator`).
- Rebuilt classes aren't in the frozen `available_neural_mass_models` snapshot,
  so `Dynamics.ontology` returns `None` → `model.ontology`-based export paths
  (dynamics.py:2285/2334) can crash.
- Destroy-before-import + `except Exception: pass` can permanently drop a model;
  synonym lookups (`from_db` miss) keep stale baked instances but mark the name
  synced.
Also switch the runtime load off the deprecated `tvb-o.owl` (see this section).

## Unify experiment selection on `SimulationStudy` (dedup the two CLIs)

`tvbo run` and `tvbo workflow` both resolve `--experiment` against a study by
matching `{key, name, label, str(id)}` on the datamodel objects, then
reverse-derive `sel = id or key or name or label` to hand back to
`SimulationStudy.get_experiment(sel)` for materialisation. The `{…}` identity set
is now shared via `_common.experiment_ids()`, but the resolution block (guard →
`sel` → `get_experiment` → `die`) is still copy-pasted in both CLIs and has drifted
(`hasattr(exp,"run")` vs `hasattr(exp,"render")`; different `die` wording). Root
cause: `get_experiment` (`tvbo/classes/study.py:117`) matches by **id only**, so
each CLI must do its own 4-field match first.

Fix at the right layer:
- Broaden `SimulationStudy.get_experiment` to accept id **or** key/name/label
  (backward-compatible — id still matches).
- Add `SimulationStudy.select_experiments(selector) -> list[runtime experiment]`
  that splits the comma-list and returns runtime (`.run`/`.render`-capable)
  experiments.
- Collapse both CLIs to one call, deleting the per-CLI 4-field match, the `sel`
  reverse-derivation, and the `hasattr` guard. The comma-list *semantics* stay
  per-CLI (`tvbo run` runs all in-process; `tvbo workflow` fans into separate
  kits) — that difference is intended, not duplication.

Deferred from a `/simplify` pass because it touches `study.py` (outside the
reviewed CLI diff) and the CLIs were being actively edited. Surfaced by 3/4
cleanup agents (reuse + altitude).

## Harmonize class names with `tvboptim`

Rename `ExplorationAxis` → `Axis` and reshape it so tvbo can declaratively
specify any `Space` configuration that tvboptim supports
(`GridAxis`, `LogGridAxis`, `UniformAxis`, `DataAxis`, `NumPyroAxis`).
`Space`/`ExplorationSpace` become aliases of `Exploration`; the slot
`Exploration.space` becomes `Exploration.axes`.

Full design, rationale, file-by-file impact, and step-by-step
implementation plan: **see `dev/tvboptim_harmonization.md`**.

### Expose dynamics parameters as `net.dynamics[name].parameters`
The built tvboptim `Network` exposes model parameters at `net.dynamics.params.<P>`
(a `Bunch`), but the tvbo schema convention is `Dynamics.parameters` keyed by model
name. Two mismatches:
- naming: tvboptim `.params` vs tvbo `.parameters`;
- shape: tvboptim `net.dynamics` is a single object, not name-keyed, so
  `net.dynamics[name].parameters.P` (the natural path for multi-model networks) fails.
Needed: expose `.parameters` (alias or rename) and make `net.dynamics` addressable by
model name, so introspection/round-trip uses one convention on both sides of the
tvbo↔tvboptim boundary. (Surfaced while inspecting the Taher2019 built model —
parameters ARE there, just at `net.dynamics.params.P`.)

## Unify Exploration / Optimization / Pareto / Inference under one `Search` concept

**Status:** design north-star; Hopf_Pareto PR ships only the forward-compatible
surgical subset (`Exploration.strategy` + `objectives` + `ExplorationAxis.transform`
+ widened `Optimization.depends_on`).

Grid sweep, gradient optimization, NSGA-II, and Bayesian MCMC are the *same
shape* — *search a parameter space with a `strategy`, score candidates against
goal(s), execute with parallelism, optionally seeded by an upstream via
`depends_on`*. They differ only along: `strategy`
(`grid|random|adam|nsga2|nuts|…`), goal-type (**0** goals = sweep · **1**
objective = minimize · **≥2** objectives = Pareto · a **likelihood** = infer),
axis payload (`domain`/`values`/`distribution`), and output shape
(evaluated-set/point/front/posterior — derivable from goal-type).

**Bayesian folds in and *simplifies*:** a `Prior` *is* an `Axis` with a
`distribution:` (the `NumPyroAxis` from `dev/tvboptim_harmonization.md`), so the
standalone `Prior` class **disappears**; `Inference` becomes a `Search` with
`strategy: nuts` + a `likelihood` goal. The one principled discriminator:
`objectives:` (optimize/Pareto) **xor** `likelihood:` (infer).

Composes with the SED-ML **Task hierarchy** (`dev/Interoperability/SedML/plan.md`
§4.2.1) and **depends on** the Axis harmonization above (supplies `Axis` +
`NumPyroAxis`). Follow-up PR migrates all workflows one-at-a-time behind
read-aliases with per-workflow byte-identity re-verification (×6).

Full design, the strategy/goal taxonomy, the Inference-under-Search analysis,
the surgical-now vs. unified-later split, migration path, and risks:
**see `dev/unified_search.md`**.

## Backend-in-Metadata + Per-Task Backend Dispatch

Move backend specification from runtime arg to metadata, so each `Task` in a
`SimulationExperiment` carries its own backend. `exp.run()` executes the full
Task DAG (integration with tvboptim, bifurcation with bifurcationkit,
exploration with pyrates, …); `exp.run('jax')` / `exp.run(integration='jax')`
/ `exp.run({'main': 'jax'})` override at runtime. Sharing the YAML is enough
to know how the experiment is run; same experiment can be re-rendered or
re-run with a different backend.

**Depends on** the Task hierarchy refactor in
`dev/Interoperability/SedML/plan.md` §4.2.1 (Integration as a first-class
Task, `Algorithm` broadened, deprecated read aliases for legacy slots).
Backend-in-metadata work is additive on top of that schema.

### Design decisions (locked)
- **Slot on Task** — `software: SoftwareRequirement`, peer to `execution:
  ExecutionConfig`. Mirrors experiment-level peers
  (`environment: SoftwareEnvironment`, `execution: ExecutionConfig`).
  YAML alias: `backend:`.
- **Polymorphic authoring form** — bare renderer key (`tvboptim`), package
  IRI (`tvbo:TVB-Optim`), or full `SoftwareRequirement` object. All coerce
  to canonical `SoftwareRequirement` via one resolver shared by YAML
  coercion and runtime overrides.
- **Resolution precedence** at run time:
  1. runtime override (`exp.run(...)`) — kwarg form (`integration=`,
     `continuation=`) and dict-by-name form (`{'main': 'jax'}`); positional
     `exp.run('jax')` overrides Integration tasks only (back-compat).
  2. `task.software` (authored or enriched).
  3. `tvbo/database/defaults.yaml` keyed by Task class
     (`Integration: tvboptim`, `Continuation: bifurcationkit`,
     `Exploration: pyrates`, …). Overridable per-install via env var.
  4. `experiment.environment` is **not** a source — it's a *constraint*;
     validated at `exp.run()` entry, `strict=True` by default.
- **Renderer-key ↔ SoftwarePackage map** — new multivalued slot
  `provides_format: [string]` on `SoftwarePackage`. The database YAMLs
  under `tvbo/database/software/*.yaml` become load-bearing (one package
  may provide many formats; e.g. PyRates → `pyrates`, `pyrates-yaml`,
  `pyrates-bifurcation`).
- **Execution model** — `exp.run()` runs **all** tasks in topological
  order of `depends_on`/`simulates`; returns `WorkflowResult` keyed by
  task name. Fail-fast by default; `continue_on_error=True` for batch.
  (Lazy variant — `exp.run(lazy=True)` returning a deferred
  `Workflow` — proposed for later, not in scope here.)
- **Execution adapters** — `Executor` per language in `tvbo/run/`:
  `PythonExecutor` (in-kernel `exec`), `JuliaExecutor` (juliacall,
  in-process), others as needed. Snakemake/Nextflow handoff is an
  opt-in render target (`exp.export_workflow('snakemake')`), not the
  default execution mode.
- **Render API** — rename `render_code()` → `render()`; default groups
  tasks by output language (one file per language). New `language` slot
  on `ExportFormat` reflects the rendered code's language (not the
  underlying engine's). Same override forms as `run()`; `task=` escape
  hatch returns a single string. Rendering does not mutate the spec.
- **Post-run enrichment** — `task.software` is enriched **in place** with
  resolved `version_spec`/`hash` from the live env after `run()`. YAML
  evolves: authored short form round-trips pre-run; enriched form on
  post-run save. Untouched tasks stay in authored form. Strict pin
  (`==X.Y.Z`) errors on env mismatch; constraint (`>=X`) satisfies-and-keeps;
  `exp.run(env='current')` forces re-enrichment.
- **Unknown backends** — hard error with did-you-mean suggestion
  (Levenshtein over registered renderer keys + package IRIs).
- **Migration** — load-time shim accepts deprecated slots (`integration:`,
  `explorations:`, `continuations:`, `algorithms:`, `optimizations:`,
  legacy `dynamics:`/`network:`/`coupling:`) and constructs equivalent
  `tasks:` list in memory with a `DeprecationWarning`. CLI command
  `tvbo migrate <yaml>` rewrites in place. Shim removal target: two
  release cycles after ship.

### Implementation plan (dependency order)
1. **Schema additions** (LinkML, in `schema/`):
   - `Task` hierarchy from SED-ML plan §4.2.1 (`Task`, `Integration`,
     `Exploration`, `Optimization`, `Continuation`, `Algorithm`,
     `ParameterTuning`, `Analysis`, `Inference`, `Surrogate`).
   - Add `software: SoftwareRequirement` slot on `Task`, with
     polymorphic `any_of: [string, SoftwareRequirement]` coercion. Alias
     `backend`.
   - Add `provides_format: [string]` slot on `SoftwarePackage`.
   - Add `language: string` slot on `ExportFormat` (registry-side, not
     schema).
   - Populate `provides_format:` on existing
     `tvbo/database/software/*.yaml` (TVB-Optim → tvboptim; TVB → tvb;
     PyRates → pyrates, pyrates-yaml, pyrates-bifurcation; jaxley → jax;
     BifurcationKit.jl → bifurcationkit; AUTO-07p → pyrates-bifurcation
     companion; …).
2. **Defaults file** `tvbo/database/defaults.yaml` — Task class →
   renderer key. Env-var override (e.g. `TVBO_TASK_DEFAULTS=/path`).
3. **Resolver** `tvbo/run/resolve.py` — single function consumed by
   YAML coercion (Task.software) and runtime overrides
   (`exp.run(...)`). Polymorphic input → canonical
   `SoftwareRequirement`. Handles the (a)/(b)/(d) precedence chain and
   the (c) constraint validation.
4. **Executor adapters** under `tvbo/run/`:
   - `python.py` — in-kernel `exec`.
   - `julia.py` — juliacall, in-process.
   - Wire each `Executor` to its language; `WorkflowResult` aggregator
     handles cross-task dependency by passing upstream `TaskResult`s.
5. **`SimulationExperiment.run()` rewrite** — topological execution
   over `tasks:`; dispatch per-task through the resolver + Executor;
   replace the `if format == ...` chain with registry-driven dispatch;
   keep the old `format=` kwarg as a back-compat alias for the
   positional override.
6. **`SimulationExperiment.render()` rewrite** — rename from
   `render_code` (keep alias); group tasks by `ExportFormat.language`;
   per-task `software` resolution; one file per language; `task=` single
   render escape hatch.
7. **Post-run enrichment** — `Executor.execute()` returns the resolved
   `SoftwareRequirement` (with `version_spec`/`hash`); `run()` writes it
   back into the in-memory `task.software`. `exp.save()` serialises the
   enriched form. Add `exp.run(record_provenance=False)` to opt out.
8. **Migration shim** — load-time read of deprecated slots in the
   pydantic `SimulationExperiment` (`model_validator(mode='before')`),
   emit `DeprecationWarning`. Per the SED-ML plan, the slots already
   exist as `deprecated:` aliases — just wire the rewrite logic.
9. **CLI migration command** `tvbo migrate <yaml>` — rewrite legacy
   YAMLs in place to the new `tasks:` form. Idempotent.
10. **`exp.export_workflow('snakemake')` / `'nextflow'`** — opt-in
    render target, not on the run path. Defer.
11. **Tests** — fixture: migrate `tvbo/database/experiments/bifurcation/
    JansenRit-bifurcation.yaml` (mixed Continuation tasks + implicit
    warmup Integration); round-trip authored ↔ enriched form; verify
    `exp.run()` end-to-end with mixed Python+Julia backends; verify
    `exp.run('jax')` overrides Integration only.
12. **Docs** — update `docs/Usage/SimulationExperiments.qmd` to show:
    authoring `tasks:` with per-task `backend:`; `exp.run()` /
    `exp.run('jax')` / `exp.run(integration=...)`; provenance enrichment
    on save; the `tvbo migrate` flow.


## Revisit Heterogenous parameter specification

- What is the current status of Network.nodes and where are the parameters to be expected?
  - Node.dynamics.parameters or Node.parameters?


## Unified `Coupling.function.rhs` with backend pre/post split

**Status:** future enhancement — purely additive, fully back-compatible.

**Motivation.** The current `Coupling` block carries three coupled
declarations:

- `local_states: [S]` — which source state vars the coupling reads
- `pre_expression: { rhs: ... }`  — per-source-node transform (pre-synaptic)
- `post_expression: { rhs: ... }` — post-aggregation transform (post-synaptic)

The pre/post split is **biologically sound** — it mirrors pre-synaptic
vs. post-synaptic processes (NT release / receptor binding /
postsynaptic potential), so it stays. But `local_states` is redundant
given sympy: any expression `parse_expr(rhs)` already exposes its
`free_symbols`, which the resolver can classify against the existing
namespace (`coupling.parameters`, `dynamics.state_variables`,
`network.parameters`, etc.). The `_i` / `_j` index convention (or
matrix form `W @ S`) distinguishes local vs. incoming nodes natively.

**Proposal.** Add an optional `Coupling.function.rhs` slot that
expresses the *full* coupling math in one sympy expression. At codegen
time the backend:

1. Parses the unified rhs (sympy `parse_expr`).
2. Detects the `Sum(W[i,j] * f(S[j]), (j, 0, n-1))` pattern (or matrix
   equivalent `W @ f(S)`).
3. **Auto-derives** `pre_expression = f(S[j])` and
   `post_expression = g(aggregate)` from the structure, with the
   aggregation step (matmul / einsum) inserted between them.
4. Keeps both forms **synchronised**: changes to the unified rhs
   regenerate pre/post; changes to pre/post can be re-composed into
   the unified rhs.

The user writes whichever form is more natural:

- Mathematically minded users → unified `function.rhs`, pre/post
  derived.
- Biologically minded users → explicit `pre_expression` (pre-synaptic
  transform) + `post_expression` (post-synaptic transform) +
  `local_states` (interface declaration).

Both forms produce identical lambdified code; the backend treats them
as alternate representations of the same coupling.

**Back-compatibility.** Existing experiments that use `local_states` +
`pre_expression` + `post_expression` continue to work unchanged. New
experiments may opt into `function.rhs` for conciseness. Coupling
blocks that set *both* the unified form and pre/post: validate-as-
equivalent at load time; raise a clear error on mismatch.

**Bonus.** `local_states` becomes optional everywhere — when omitted,
it is auto-inferred from `pre_expression.rhs` (or `function.rhs`) via
sympy free-symbol analysis. Same back-compat story.

**Scope.** Two days of schema + codegen work, plus tests on the
existing experiment YAMLs (RWW, JR, EI-Tuning) to verify zero
regression. Not blocking the RC paper — flagged here for after that
schema PR lands.


## Operating-point primitive: constraint-defined parameters + linear-response observations

**One capability, driven by Deco 2014 Figs 2c/5/6, but use-case-independent.** The unifying
insight: FIC's `J_i` and the Fig 5/6 analytic observables (covariance, PSD, Fisher) are *the same
kind of thing* — **symbolic quantities defined by a relation, resolved at a computed operating
point**. Build the primitive once; both fall out.

### First-class `OperatingPoint`
- A declared, resolvable fixed point of the (deterministic) vector field:
  `operating_point: {method: time_integration | newton}`. Reuses the existing
  `initial_state: time_integration` machinery; `newton` solves `f(x*,p)=0` directly (noise off).
- BOTH free-parameter constraints (below) AND observations (below) are evaluated **relative to
  it**. It is the shared object, resolved per-backend (Julia time-integration for continuation;
  JAX `solve_fp`/newton for observations).

### FIC = constraint-defined `DerivedParameter` (NO control law)
- `J_i` is already declared `free: true` (a *derived* parameter — value determined by tuning, not
  the user). Its set-point is already declared as the FIC `Algorithm`'s `TuningObjective`
  (`activity_target`, `target_variable: S_e`, `target_value`). **Reuse both — zero new schema.**
- Continuation realization: extend the **network-continuation** codegen (`_build_network_context`
  + the `network_mode` branch already built) so `free` params become **extra unknown blocks whose
  defining equation is the `TuningObjective` residual** (`H_e − target = 0`), not an ODE. This is a
  defining-function / extended-system continuation. `J_i(G)` falls out of the equilibrium; the
  Deco "FIC diverges at G>4.45" appears as a **feasibility-boundary bifurcation** of the
  constrained system. Faithful to Deco's `fsolve` (algebraic tuning), no fake timescale, no
  spurious slow eigenvalue.
- Generalizes for free to EIB `wLRE`/`wFFI` and any future tuned parameter.
- **Parked (separate, optional):** a genuinely *dynamical* homeostasis mode — the same law as a
  co-integrated slow state `dJ_i/dt = η·S_i·(H_e−ρ)` (Vogels/Schirner live plasticity). Same
  equilibrium; build only when a live-plasticity model needs it. NOT for faithful Deco.

### Operating-point-relative observations (Figs 5/6) — closes E1/E2
- Observation callables must receive **(resolved model params, operating-point state,
  `network.weight`)** as **symbols via `render_expression`** (`tvbo/codegen/code.py:1216`), jnp
  output — NOT `numpy`, NOT `jnp.array(repr(...))`. **Supersedes the reverted numpy
  `_network_weights` block** in `tvbo-tvboptim-observation.py.mako` (working-tree revert pending).
- Runs on **tvboptim/JAX** (host-observation path — the analytic observables are differentiable
  linear-response quantities at the FP; see `reference-tvbo-observation-host-grid-split`).

### Symbolic solver spec (backend-abstracted printer primitives)
**This is the linear-algebra layer for the observables above — declare symbolic, emit per-backend,
implement only the backend a use-case needs.** Follows
`feedback-codegen-backend-abstracted-printer-primitives` (cf. `~/tools/tvbo-arrayops`).
- **Jacobian `A` — fully symbolic, no autodiff.** The dfun is already symbolic, so
  `A = sympy.Matrix(dfun_rhs).jacobian(state_vars)`; print via `render_expression`. Backend-
  independent by construction (emits `jnp` on JAX, `ForwardDiff`/analytic on Julia). Noise `Q`
  from the declared noise (`nsig`/`sigma` → diagonal), also symbolic.
- **Solve/inverse primitives** — the only true numerical-linear-algebra ops. Add as ONE
  sympy-printer handler each, dispatching to per-printer primitives:
  - `lyapunov(A, Q)` → continuous Lyapunov `AΣ+ΣAᵀ+Q=0` (covariance, Eq 24).
    JAX: `jax.scipy.linalg.solve_continuous_lyapunov`. Julia: **stub** (add when needed).
  - `matrix_inverse(M)` → for PSD `(iωI−A)⁻¹ Q (iωI−A)⁻ᴴ` (Eq 28), swept over ω.
    JAX: `jnp.linalg.inv` / `jnp.linalg.solve`. Julia: **stub**.
  - (Fisher Eq 33/34 = trace/quadratic forms over the above — reuse `matrix_inverse` + existing
    reduce/trace primitives; no new primitive.)
- Implement the **JAX emission now** (Figs 5/6 target JAX); leave Julia emission stubbed with a
  clear `NotImplementedError` so the general structure exists without gold-plating an unused
  backend. Register the primitives in the arrayops printer table, not ad hoc in the template.
- **Definition of done:** `render_expression` of a Lyapunov/PSD relation over the symbolic Jacobian
  emits valid jnp; the numbers match the current `deco2014_plot.py` `solve_fp`/moments/PSD/Fisher
  values (bit-close), with the whole thing declared in the recipe — no hand-rolled plotter math.

### Cleaner Fisher (deferred; current `lr_fisher` is the forward-only interim)
Status: `lr_fisher` (`_linear_response.py.mako`) is SHIPPED — a self-contained partial that sweeps the
stimulus event's ΔI, re-settles the operating point (settle + Newton polish) per ΔI, and builds
`FI = μ'ᵀP⁻¹μ' + ½Tr[(P'P⁻¹)²]` by **finite differences**. It is metadata-driven (regions/variable/ΔI
from the `locc_stim` event + `analysis: {type: fisher}` params), not a use-case hack, and sits at the
same level as `lr_covariance`/`lr_psd`. Kept as-is by decision (2026-07). Two clean-ups when revisited:
- **Factor a general `moments(θ)` primitive** — μ(θ), P(θ) at a swept parameter — reusable for *any*
  parametric analysis of the linear-noise moments (sensitivity, identifiability, …), with `lr_fisher`
  becoming a thin reduction on top. Removes the sweep/re-settle duplication between cov/psd and Fisher.
- **Differentiable Lyapunov ⇒ autodiff Fisher.** The finite-difference scheme is a *direct consequence*
  of the forward-only covariance choice (JAX eig has no eigenvector gradient, #2748). Add the
  `custom_vjp` adjoint-Lyapunov (`AᵀS+SA+P̄=0` → `Ā=SPᵀ+SᵀP`, `Q̄=S`; validated exact to 5e-7 vs FD)
  so μ′, P′ come from `jax.jacobian(moments)(θ)` — no sweep bookkeeping, closest to a symbolic FI
  reduction, and makes the whole moments stack differentiable/optimisable. Reopens the forward-only
  decision, so gated on a use-case that needs gradients through the moments.

Full narrative dev-plan + phase gates:
`…/replication_studies/Deco2014/docs_Replication_Deco2014/DEV_PLAN_recipe_native.md`.

## Unify by-label node reconcile as a backend-printed arrayops primitive

**Status:** someday / generalization (deferred 2026-07-15). Backend-portable **indexing**,
not symbolic math — sympy is only the delivery vehicle (printer primitive), the op is a gather.

Every by-label node alignment in tvbo computes a gather index in Python (`region_alias_map()`
+ shared-labels) and applies it ad hoc:
- dataset-FC reconcile — `resolve_dataset_observations` / `dataset_reconcile_index` (`.sel` / numpy take),
- the `from_experiment` free-parameter warm-start — `estimate__<param>` → reconciled seed
  (reuses the dataset reconcile machinery today; see `dev/from-experiment-parameter-warmstart-design.md`),
- any future node-map / atlas-crosswalk alignment.

Generalize into ONE backend-abstracted arrayops primitive
`gather_by_label(arr, src_labels, dst_labels, alias_map)` (vector → 1 axis; square matrix → both
axes), registered in the arrayops printer table (numpy / jax / julia), so all three share it.
**Not needed for the warm-start itself** — that seed is a resolve-time constant, so the existing
Python reconcile is reused. Worth doing when a second consumer needs a *backend-portable*
label-gather (inside the emitted/differentiable path), or purely to dedup the three ad-hoc paths.
Register in the arrayops printer table, not ad hoc in a template
(`feedback-codegen-backend-abstracted-printer-primitives`).

## Improve Observations

- **Stateful Observations** (BOLD / balloon-Windkessel pattern). Some
  observers carry their own ODE state (s,f,v,q) driven by parent
  neural activity, with a linear/algebraic read-out applied to the
  integrated state. Decision: extend `Observation` with an optional
  embedded `dynamics:` (referenced by IRI to a regular `Dynamics`
  entry such as `BalloonWindkessel.yaml`), an `input:` slot
  (`source` variable on parent + `forcing_slot` name on the
  sub-Dynamics), and a `readout:` algebraic map. Also add
  `Dynamics.forcing_inputs: [name]` to distinguish external driving
  inputs from network coupling. Preserves Observation semantics (no
  feedback, no role in `Network` topology) while reusing the full
  Dynamics infrastructure. Full schema sketch in
  `dev/Interoperability/vbjax.md` §7.9. Chains naturally with the
  pipeline DAG below via the same `input:` field.

- If we change to the task-based approach (SED-ML interoperability), we might also define the Observation-Pipeline as DAG of Tasks. By that, it gets more aligned with the other Specs (e.g. Pipeline).
    - A tasks requires specification of input, function, output (like FunctionCall).

- Is DerivedObservation as concept really sound? Actually, it would be more minimal to use single Observation class, but to clarify what is the output dimensionality.
  - Is it still time-series or has it changed dimnensionality, i.e. was as a dimensionality reduction on time-dimension applied

So we need to find a generalizable way to describe these Observations as pipeline of tasks, what needs to be done to derive a certain observation from the raw timeseries.
- Is additional data needed (external observation)?
- What dimension is looked at?
- What data is selected?

Examples:
There are classical examples, however we need to find a solution to describe any potential Observation

- Mean timeseries
- Frequency Spectrum / Single Band-Power
- Correlation (FC, FCD)
- Projection (Matrix-Multiplication)
- Convolution


### Compile the algorithm tuning loop with `lax.scan` (GPU + differentiable)

**Deferred 2026-07-13** — investigated, scoped, and consciously postponed (not a
CPU win; do it when we go GPU or want gradient tuning).

**Motivation.** `run_<algo>` in `tvbo-tvboptim-algorithm.py.mako:583` runs the
tuning iterations as a **Python `for i in range(n_iterations)`** that calls the
jitted `model_fn` per step. Each iteration = one `simulation_period` (720 ms = 1
BOLD TR) sim + one parameter update; `n_iterations` is the tuning-step count
(50 000/stage for the Schirner 6-stage schedule). Converting the loop to
`jax.lax.scan` (a *fold* — it compiles the sequential chain, does not parallelise
it) would (a) run the whole tuning chain as **one on-device kernel** (no per-step
Python↔XLA dispatch/host-sync), unlocking real GPU throughput, and (b) make the
fit **differentiable** (gradient-based tuning / Optax, `gradient_eib`).

**Why deferred (evidence).** By first principles the CPU payoff is small: each
iteration is dominated by the jitted 720-step × N-node `model_fn`; the ~10 extra
jax ops (buffer roll, reducer evict/add/emit, update rules) are **async-dispatched**
(the Python loop runs ahead of the device) and the only forced host-syncs are the
periodic `print` and the end — so removable overhead is a few % on CPU. A wall-clock
measurement was attempted but the box was at load ~54 (99 python procs from other
sessions), so timing was unusable; the first-principles argument stands. Subject
parallelism is **not** blocked by this — it lives at the `tvbo workflow` / Snakemake
layer (one job per subject, own FC path). So this is a **GPU + differentiability**
enhancement, worthless for the current CPU + Snakemake replication path.

**Scope when picked up (large, high-risk — gate + byte-equivalence test).** The
carry must thread: `state` (`initial_state.dynamics`, `noise.key`,
`coupling.*.wLRE/wFFI`, `dynamics.J_i`), the sliding-window **buffer**, the
**streaming-reducer accumulator** + its periodic exact `resync`, and the **BOLD
monitor** `_history` (functional `eqx.tree_at`, no in-place mutation). Hard parts:
`result_history.<param>.append(...)` records the **full N×N `wLRE`/`wFFI`** at
`save_every` — `scan` emits every step or none, and every-step N×N over 50 000 is
~28 GB/param, so preserving matrix snapshots needs a **chunked scan** (scan in
`save_every` blocks, record matrices between blocks). Also: collectible-observation
appends, `print` → `jax.debug.print`, and the **nested-include** mode
(`run_<inner>()` inside the loop → nested scan; Schirner uses `combined` so N/A).
Companion to the streaming-reducer work below (same loop body). MUST ship behind a
codegen flag with the Python loop as fallback + a `scan == python-loop`
byte-equivalence test before it becomes default.

### DONE (2026-07-13): equation-based derived observations

Added a general **`first_passage` aggregation** (`AggregationType` enum +
`tvbo-tvboptim-observation.py.mako` branch reading `parameters.threshold`) — first
sample where a source crosses a threshold over the time axis, via `argmax` of the
crossing (backend-independent). Works end-to-end (Schirner Exp 50 DM decision:
`t_A`/`t_B` = first 40 Hz crossing of A/B_PFC).

**Fixed** the derived-observation gap: `compute_all_observations`
(`tvbo-tvboptim-experiment.py.mako`) only emitted CALLABLE/function-based derived
observations — an `equation`-pipeline stage set `pipeline_call=None` and was
silently dropped, so `obs.<name>` was never set (`'Bunch' object has no attribute
<name>`). Added an `elif pipeline_equation and src_obs_list` branch that binds each
source observation to a local and renders the equation inline via `jaxcode`
(= `render_expression`, so `user_functions` + local `parameters` are forwarded,
backend-independent). Covers BOTH the base run and the exploration grid (shared
function). `import functools` added (render_expression emits `functools.reduce`
for `Min`/`Max`). Validated byte-exact: Schirner Exp 50 `integration_time`/`winner`
over scalar first-passage sources, AND a synthetic `obs_v+obs_w` / `2*(obs_v-obs_w)`
over (300,1,1) TIME-SERIES sources (max err 0.0). No regression (observation +
experiment codegen suites green). The gap was tvboptim-specific — Julia/pyrates/report
templates don't share this path.

### Workflow from_experiment dependency: SLURM afterok not wired

FIXED (2026-07-14) the Snakemake key-mismatch: `plan().depends_on` recorded
`str(int(source.id))` while `_ep_by_key`/output dirs key by `_san(experiment_key)`
(explicit `key` if set) — an explicit-key source silently dropped its edge, and a
non-numeric ref crashed `int()`. Now `plan()` stores the raw id/key/name (no int),
and `_emit_snakemake_study` resolves each dep to the source's sanitized key via an
id/key/name→key map (`_key_of`), so the emitted `input:` points at the source's
real output dir. Validated: dep to a `key:baseline` (id 30) source →
`input: f"{OUT_DIR}/baseline/result.h5"`; 37/37 workflow tests pass.

STILL OPEN: the `depends_on` comment promises "SLURM afterok", but only
`_emit_snakemake_study` consumes it — no SLURM emitter turns a cross-experiment
`from_experiment` edge into an `afterok` (the existing afterok is the within-experiment
array→finalize gather). A study with a `from_experiment` dep submitted via SLURM runs
dependents unordered. Wire cross-experiment afterok into the SLURM path (or drop the
comment's promise).

### first_passage could emit a time, not a sample index (altitude)

`first_passage` currently returns a sample INDEX; consumers convert to a time by
multiplying by the sampling step (Schirner Exp 50: `Min(t_A,t_B) * 0.5`, where 0.5
duplicates `integration.step_size`). first_passage is the only aggregation whose
output unit isn't the source's unit, and the literal drifts if step_size changes.
Deeper form: have the aggregation emit `argmax(...) * dt` (a first-passage time in
ms) — then `integration_time` = `Min(t_A,t_B)` with no literal and the step lives
in one place. Deferred (behaviour/contract change, and needs correct dt/period
handling when the observation subsamples): the effective step is `period` when set,
else the integration `dt`; getting that wrong silently mis-scales. Winner (order
comparison) is unaffected either way.

### Julia printer: `argmax(...) * scalar` fails

`render_expression('argmax(r >= thr) * dt', format='julia')` raises
`TypeError: can't multiply sequence by non-int of type 'Symbol'` in the Julia
printer (jax/numpy render fine: `dt*jnp.argmax(jnp.greater_equal(r, thr))`).
Surfaced 2026-07-13 while proving the DM first-passage observation (decision =
first threshold crossing → integration time) is expressible with existing
array-ops (`argmax`/`where`/`max`). Backend-independence gap — the DM decision
observation renders on jax/numpy but not Julia until this is fixed. Low priority
(DM circuit runs on jax/numpy), but it's a genuine printer bug in the
`argmax`-times-scalar path.

### Codegen-emit streaming reducers (tvbo emits, tvboptim ships none)

**Principle.** Windowed pipeline reducers (`compute_fc` → an incremental FC
accumulator, dFC/FCD, metastability) are backend-independent math. tvbo EMITS the
concrete reducer into the generated experiment; the backend provides only the
framework the reducer plugs into (tvboptim's sliding-window loop drives an
`add/evict/emit/resync` protocol). Realizes the global "tvbo extends backends via
codegen" principle. (Origin: referencing tvboptim's unreleased `windowed_cov` by
string broke CI on the pinned `0.2.4` — hand-writing a realization in one backend
was the wrong layer.)

**Stage 1 — DONE (2026-07-14).** The *algorithm* streaming path (mechanism A) now
emits the reducer instead of importing a tvboptim factory:
- `StreamingReducerSpec.factory` (tvboptim import string) → `.emitter` (name of a
  tvbo-emitted factory, `_make_windowed_fc_reducer`).
- `_factory_available` + the `importlib` guard **deleted** — no external factory
  to resolve, so the `0.2.4` CI-breakage stopgap is gone by construction.
- `tvbo-tvboptim-algorithm.py.mako` emits `_make_windowed_fc_reducer` locally (a
  `SimpleNamespace` with `add/evict/emit/resync`) carrying the **numerically-stable
  mean-centred** co-moment: Welford `add` + reverse-Welford `evict` downdate +
  two-pass `resync`. NOT the naive one-pass form → byte-identical to `compute_fc`
  incl. DC offset (8.3e-17 vs the naive 8.5e-4; proof
  `scratchpad/verify_windowed_cov.py`). So the precision issue (old finding #1) is
  fixed **by construction** — no separate windowed_cov fix needed.
- Generated EI_Tuning code: 0 `windowed_cov` references; `test_experiment_runs
  [EI_Tuning]` passes. tvboptim's `windowed_cov`/`windowed_fcd` are now
  unreferenced → **delete them from tvboptim** (its own branch); the uncommitted
  `windowed_cov` WIP mis-parked on `differentiable-delays` should be **dropped, not
  fixed**. `types/spaces.py` (Space array-axis) is unrelated — its own tvboptim PR.
- Files: `tvbo/codegen/streaming_reducers.py`, `tvbo-tvboptim-algorithm.py.mako`.

**Stage 2 — DONE: symbolic, use-case-independent, multi-backend lowering.** (2026-07-14)
- **Recipe is metadata**: the reducer is a declarative `StreamingReducerSpec` authored as
  YAML in `tvbo/database/reducers/*.yaml` (`state` + `add`/`evict`/`resync` assignment
  strings + `emit`); `streaming_reducers.py` loads and registers them. **No use case is
  hardcoded** — adding a reducer is a YAML file. Sequential reassignment encodes the data
  flow (no `new_*` temporaries).
- **General resolver**: `tvbo/codegen/reducers.py::resolve_streaming_reducer(spec, fmt)`
  parses any spec's strings to sympy against its vocabulary and prints per backend. No
  reducer-specifics.
- **General template**: `tvbo-tvboptim-algorithm.py.mako` iterates the resolved
  state/assignments into backend scaffolding with generic comments (0 FC-specific words).
- **Printer array-ops**: `outer`/`diag`/`zero_diagonal`/`matmul` added to the printers +
  parse vocabulary (`expression.py`); Julia `global_mean` (axis-mean) wired. The SAME spec
  now lowers cleanly to **jax and julia** (byte-identical to `compute_fc` incl. DC offset).

Remaining Stage-2 follow-ons:
- **Julia streaming *path*** (not just the reducer math): needs a Julia algorithm template
  that emits the generic scaffolding + `using LinearAlgebra`/`Statistics`, and a Julia
  runtime to validate. The reducer math is ready.
- **Unify with `Observation.dynamics`/`resolve_reduction`** (extend that declarative-
  recurrence engine with sliding-window `evict`) so there is one reduction engine, not two
  — gated on that WIP.
- FCD (`compute_fcd`) YAML with a `stride` emit when the template's stride branch lands
  (dropped from the registry meanwhile; `is_windowed_reducer(compute_fcd)` is now False).

**Stage 2 — FCD.** `_make_windowed_fcd_reducer` is registered (stride emit_kind)
but not yet emitted; `streaming_capable` reports False so it is inert (not broken).
The stride branch (emit per-window into a growing stream, `finalize` once) is the
documented follow-on. When wired, `finalize` must take a stacked array, not a
Python `list` (won't trace under the jitted scan).

**Validation.** Byte-identity against the recompute reference
(`TVBO_STREAMING_REDUCERS=0`) on the online-tuning experiments (EI_Tuning,
RWW_BOLD_FC). Stage 1 verified; Stage 2 must re-verify per backend.

### Explorations: Observations should be also available in Explorations

So I want to setup:
- Observation (let's say just mean)
- Exploration axis (parameter sweep)
- Plot Observation over a


## Documentation

- [ ] Always describe both, 1) Python API for model specification, 2) Pure Yaml
    > Currently, it is not clear how to use python-API, e.g. for defining/adding Observations and pipelines. Also always using pure yaml is not intuitive enough for iteratively setting up experiments.


## Migrate to Pydantic

- [ ] Find out if there is any benefit of linkml gen-python instead of gen-pydantic. Do we really need both?
- [ ] Change import of metadata-classes, so we can import all from tvbo.classes or tvbo.schema.
    > Having a common import structure for classes with extra functionality (inherited from LinkML) and pure LinkML export would be nice, so it's not confusing from where to import certain classes.


## Linkml Yaml shorthands
- [ ] Investigate shorthands for specs.
    > - It is a little cumbersome to specify equations always with `rhs` (equation={'rhs':'x+y'}), since we most of the time only need to specify `rhs` attribute. However it is relevant to have a proper `Equation` class. We need different equation types (differential, etc.), for StateVariable (ODE, PDE, ...), DerivedVariable (just algebraic), etc.
    > - So shortcutting to `equation='x+y'`, which resolves into `equation={'rhs':'x+y'}` would be really useful. But it needs to be linkml-native. No monkey-patching or hacks.
    > - It would be great that we can set for each class, that has equation as property, so we can define axioms, which equation-type they expect. We need to find out, if this is possible.



## Exploration Space must be keyed

Currently,
exp.explorations["a_sweep"].space

is list.

[ExplorationAxis({'parameter': 'a', 'explored_values': [-2.0, -1.0, 0.0, 1.0, 2.0]})]


But we want to be able to change the space of a specific axis. therefore we need keys.


## Interopearbility

## Data Standards
Neurodata without Borders (NDWB)


## Bifurcation result needs to be also xarray structure!
- selection of variables should be possible etc.


## Native result container format (single-file save/load + `StudyResult` hierarchy)
Result objects can be *exported* but not *stored/reloaded* as a self-contained container.
Consumers (e.g. replication run scripts) fall back to ad-hoc `np.savez` because there is no
native "save this result to one file, load it back" path.

- **Current state.**
    - Result types: `SimulationResult`, `ExplorationResult(Bunch)`, `AlgorithmResult`,
      `OptimizationResult`, `ExperimentResult` (in `tvbo/data/types.py`). There is **no
      `StudyResult`** — a `SimulationStudy.run()` returns a loose dict/list of per-experiment
      results with no container.
    - `ExperimentResult.export()` / `.to_bids()` write a **multi-file BIDS BEP034 directory**
      (yaml + per-observation `.nc`/`.h5`). This is **one-way** — there is no `from_bids` /
      `load_result`, so an exported result cannot be reloaded into the typed objects.
    - `experiment_result_io.py` has a fingerprint-keyed sidecar+h5 **cache** (internal), not a
      user-facing result container.
- **Needed (two tiers, one API — mirrors the network sidecar/companion design).**
    1. **Single-file, self-contained container** for when the full BIDS tree is overkill:
       `result.save(path.h5)` / `Result.load(path)` round-trip, xarray/HDF5-backed, one file per
       `SimulationResult`/`ExperimentResult`. Extension-dispatched (`.h5`/`.nc`/`.zarr`).
    2. **`StudyResult` container** at the top of the hierarchy
       `StudyResult → ExperimentResult → {SimulationResult, ExplorationResult, AlgorithmResult,
       OptimizationResult}` — serializable either as one grouped file (HDF5 groups / zarr) or a
       fan-out of per-experiment containers, with the study spec (key/title/provenance) carried
       alongside.
    3. **Round-trip load** (`from_file` / `load_result`, `from_bids`) so `export`/`to_bids` output
       reloads into the typed objects — export must not be one-way.
    4. **BIDS stays the interoperable/full tier**; the single-file container is the quick/local
       tier. `result.save()` picks the container by extension; `.to_bids(dir)` writes the tree.
    5. `ExplorationResult` must serialize its **keyed** observations + axes + `n_up` as labeled
       xarrays (per the keyed-xarray-never-positional rule), so a swept result reloads with named
       coordinates (K, node, …), not positional arrays.
- **Related:** `Harmonize SimulationResult with tvboptim's NativeSolution`, `Bifurcation result
  needs to be also xarray structure!`, `Data Standards` (NDWB) above — a unified result container
  should subsume all of these.
- **Surfaced by:** the Taher2019 replication `code/run_sweeps.py`, which currently writes
  `output/exp*.npz` as a documented stopgap pending this.


## Harmonize IRI resolution and DB metadata fetching across all classes
- Currently each runtime class handles `iri`-based sourcing differently:
    - `Coupling.__init__` auto-resolves via `_populate_from_ontology` after super().__init__
    - `DynamicalSystem.__init__` resolves via the registry before super().__init__ (loads full YAML and merges)
    - `Network` derives `name` from `iri` for `atlas` / `tractogram` but does no DB fetch
    - `SimulationExperiment.__init__` does its own backfill for nested dicts (parcellation/tractogram/atlas)
- Each class also has its own `from_db` / `from_file` / `from_ontology` factories with subtly different behavior.
- Needed: one canonical IRI resolution layer that any schema class can opt into:
    1. Single `_resolve_iri(iri, category) -> dict` helper (registry-first, ontology fallback).
    2. Consistent rule for "iri given, name missing/default" → load and merge (user kwargs win).
    3. Apply uniformly to Dynamics, Coupling, Tractogram, Parcellation/BrainAtlas, and any future class with an `iri` slot.
    4. Drop duplicate ad-hoc backfill code in `SimulationExperiment.__init__` and `Network.__init__` once the per-class path is uniform.


## Backend-independent declarative network construction (`NetworkGenerator`)

**Motivation.** Today procedural networks are built via
`graph_generator.builder: Callable` pointing at a Python function
(see `dev/Replication/Koller2024/.../koller2024_networks.py`). This is
the Python-only entry point. To make TVBO truly backend-independent,
the construction spec itself should live in the YAML so a Julia or
MATLAB codegen can emit equivalent native code without porting the
Python builder.

**Terminology.** Use `NetworkGenerator`, the standard graph-theory
term — NetworkX has `networkx.generators`, igraph has
`Graph.Generators`, Graphs.jl has `SimpleGraphs.Generators`. A
generator yields a `Network` (`weights`, optionally `lengths`,
optionally per-node parameters). The generator is metadata on
`Network`, not a parallel concept.

### Execution engine — generic procedure interpreter (decision: locked)

> **Detailed design: [`dev/GenericProcedureEngine.md`](dev/GenericProcedureEngine.md).**

The two tiers below define the **schema** (what a generator spec looks
like). The **engine** that executes those specs is the piece that
decides whether `tvbo/graph_generators/builtins.py` survives. Decision
(reservoir-computing review): a dynamics RHS, a generator `procedure:`,
and a `Distribution` are all the *same kind of symbolic spec* and must
be interpreted by **one** engine — the existing sympy → `JaxPrinter` /
`JuliaPrinter` pipeline in `tvbo/codegen/code.py`, extended with the
procedural primitives (`sample`, `eigvals`, `pairwise_distance`,
`stochastic_mask`, `normalize`, …) and a distribution → backend-sampler
map (vocabulary aligned to numpyro / `Distributions.jl`:
`Normal`/`Uniform`/`LogNormal`/`Beta`).

- **Stage 1 (now):** per-generator Python in `builtins.py` stays as a
  *demoted* numpy realisation; the symbolic `procedure:` in each
  `tvbo/database/graph_generators/<Type>.yaml` is authoritative.
- **Stage 2:** build the generic procedure-DAG evaluator (eager numpy
  mode), migrate `RandomReservoir` / `WeightShuffle` to pure-YAML
  `procedure:`, and **delete `builtins.py`** — new generators become
  pure YAML. The `bindings.python` / `builder: Callable` path survives
  only as the rare, explicitly-flagged library-wrapper exception.
- **Stage 3 (only if needed):** emit procedures into backend source for
  on-device / per-trial / differentiable generation.

Open question to resolve before Stage 2: cross-backend RNG
reproducibility contract (numpy PCG64 ≠ jax Threefry ≠ Julia RNG) — see
the design doc §4.

### Two tiers of generator

**Tier 1 — named built-in generators (small, finite catalogue).**
Most papers and most vbjax examples just need one of:

- `Complete {n}` — all-to-all (vbjax `00_intro`, `parsweep`, `01_sweep`)
- `Grid2D {nx, ny, extent}` — Cartesian sheet
- `SphericalGrid {nlat, nlon}` — drives neural-field examples
  (vbjax `make_shtdiff`, classical TVB surface models)
- `Lattice3D {nx, ny, nz}` — cubic lattice
- `FromAtlasCentres {atlas}` — node positions from a parcellation
- `ErdosRenyi {n, p}` / `BarabasiAlbert {n, m}` / `WattsStrogatz {n, k, p}`
  — standard random-graph models
- `FromTractogram {source_file, format}` — loaded connectome
  (already the dominant TVBO path — this just names it as one
  generator among several)

Each is a named subclass of `NetworkGenerator` with explicit slots; no
DAG, no expressions. Schema cost: tiny.

**Tier 2 — `Procedural` generator (DAG of construction steps).** For
papers whose network construction *isn't* one of the named built-ins
(Koller2024 2D-sheet with stochastic masks + Gaussian fields +
gradient template is the motivating case), allow a `Procedural`
generator whose body is an ordered DAG of named intermediates:

```yaml
network:
  generator:
    name: Koller2024_2DSheet
    type: Procedural
    layout:
      type: Grid2D
      nx: 30
      ny: 30
      x_extent: {value: 140.0, unit: mm}
      y_extent: {value: 140.0, unit: mm}
    derived:
      d_ij:
        type: pairwise_distance
        metric: euclidean
        positions: layout
        self: inf
      a_ij:
        type: equation
        rhs: "(1 / (2*sigma)) * exp(-d_ij / sigma)"
        parameters: {sigma: 10.0}
      mask_ij:
        type: stochastic_mask
        condition: "d_ij <= abs(sample)"
        sample: {distribution: Exponential, scale: 17.0}
        seed: 42
      a_masked:
        type: equation
        rhs: "a_ij * mask_ij"
      a_normalized:
        type: normalize
        axis: 0     # column-normalize
        of: a_masked
      sink_pdf:
        type: gaussian_pdf
        positions: layout
        mean: [40.0, 40.0]
        cov: 300.0
      source_pdf:
        type: gaussian_pdf
        positions: layout
        mean: [100.0, 100.0]
        cov: 300.0
      gradient_template:
        type: minmax_rescale
        of: "sink_pdf - source_pdf"
        to: [-1, 1]
    outputs:
      weights:
        type: equation
        rhs: "transpose(a_normalized * (alpha * gradient_template + beta))"
        parameters: {alpha: 2.0, beta: 4.0}
      lengths:
        type: equation
        rhs: "transpose(d_ij)"
        diagonal: 0.0
      node_parameters:
        gradient_template: gradient_template
```

**Schema for `Procedural`:**

- `layout`: a `NetworkGenerator` from Tier 1 that produces positions
  (typically `Grid2D`, `SphericalGrid`, `FromAtlasCentres`).
- `derived`: an ordered DAG of named intermediates, each one of:
  - `pairwise_distance` (with metric)
  - `equation` (sympy/numpy expression over previously-named
    intermediates and parameters)
  - `stochastic_mask` (Boolean mask from a distribution sample
    against a condition; seed-controlled)
  - `gaussian_pdf` / `distribution_pdf` (evaluate a distribution
    PDF at the positions)
  - `normalize` (axis-aware scalar normalisation: column/row/global)
  - `minmax_rescale` (with target range)
  - `reduce` (sum/mean/max along an axis)
- `outputs`: required `weights`, optional `lengths`, `node_parameters`
  mapping each node-shaped intermediate to a named `Node.parameters`
  entry.

**Why generalizable, not Koller-specific.** Every primitive above is a
textbook graph-construction operation: distance kernels, threshold
masks, Gaussian fields, axis normalisation. Roberts 2019, Pang 2023,
Cabral 2011 all assemble networks from the same vocabulary.

**Codegen targets.** Each backend in `tvbo/templates/` (currently
`tvb`, `tvboptim`, the report templates, the Julia template under
`tvbo-nd-experiment.jl.mako.py`, and the future `vbjax` template)
emits equivalent native code from the same generator spec:

- Python (tvb / tvboptim / vbjax): numpy + scipy.spatial /
  scipy.stats / jax.numpy.
- Julia (Graphs.jl + Distributions.jl).
- MATLAB (built-ins).

**Migration path.** The existing `graph_generator.builder: Callable`
path stays — generators are an additional, more declarative route. A
generator with no Callable invokes pure-codegen materialisation; a
Callable with no generator stays Python-only (current state). Studies
can mix: prefer named Tier-1 generators when applicable, fall back to
`Procedural` for paper-specific constructions, fall back to a
Callable for one-off Python logic.

**Scope.**
- Tier 1 (named built-ins): small, ~1 day per generator. `SphericalGrid`
  is the only one needed by vbjax (drives the neural-field examples).
- Tier 2 (`Procedural` DAG): 2–3 days of schema + codegen work + tests.

**Defer Tier 2 until** after the Koller replication ships its first
end-to-end run, then come back to migrate Koller's 2D sheet from
`koller2024_networks.py:build_2d_sheet` to a pure-YAML `Procedural`
generator and delete the builder module entirely. Tier 1 can land
incrementally as needed by individual interop plans.


## Small-scale simulator backends: shared lowering core + native Brian2

Turn the one working small-scale backend (NeuroML/LEMS) into a **shared core +
thin backends** layer so Brian2 (native, next) and later NEST/Jaxley plug in with
zero duplicated network-lowering or synapse logic. Full plan + paste-ready handoff
prompt: `dev/Interoperability/Small-Scale-Simulators/Small-Scale-Adapter-Layer-Plan.md`
and `…/HANDOFF-SmallScale-Core-and-Brian2.md`. Motivation (July 2026 Deco port):
NeuroML is a *specification/interchange* standard, not a network *execution* one —
jNeuroML→Brian2 refuses networks, →NEURON/→NetPyNE error, EDEN 0.2.3 rejects
`<Attachments>`/`select-reduce`, and jLEMS runs the 71,640-conn Deco column at
~190 s per 100 ms. So TVB-O must own its fast small-scale backends, via one shared
core (`tvbo/adapters/smallscale/lowering.py`) — not N copies of the lowering now
welded inside `neuroml.py`.

**Phase 1 (extract shared core, refactor NeuroML onto it, behavior-preserving)
MUST also close these findings from the 2026-07-21 pre-commit review of the
NeuroML diff** — they live in `_build_network_context` only, so the extraction
brings the duplicated `_build_std_network_context` (std-types path) up to parity
for free:
- [ ] **#1 (highest):** id-collision (`_unique_component_id`), ComponentReference
      satisfaction (Poisson background), and per-connection dedup exist ONLY in
      `_build_network_context`, NOT `_build_std_network_context`. A *standard-types*
      network with a repeated input or a `poissonFiringSynapse` background still
      hits the bugs already fixed for the custom path. Route BOTH builders through
      the shared core.
- [ ] **#8:** event-source `dyn_params = _normalize_edge_params(...)` is recomputed
      per-node inside the loop (both builders); hoist in shared `build_populations`.
- [ ] **#3:** `input_components` is a new context key only `_build_network_context`
      produces; the template's `net_ctx.get('input_components', [])` silently emits
      no inputs if a context lacks it. Return it uniformly; drop the silent fallback.
- [ ] **#6:** the input-ComponentReference type→instance redirect works only because
      `input_components` aliases the same `params` dict as `inputs`. Make it explicit.
- [ ] **#4/#5:** the custom-synapse edge-override overlay `copy.copy(param);
      _p.value=…` mutates a LinkML object (writing-models footgun) and nulls the
      param if the override dict lacks a `value` key. Construct a fresh Parameter;
      guard `if "value" in _pinfo and _pinfo["value"] is not None`.
- [ ] **#2:** per-connection `weight` is applied only to a DerivedVariable literally
      named `i`; a conductance synapse inheriting `i` from its base would lose weight
      silently. In Brian2 synapse rendering apply weight to the output current
      generically (Brian2 does this natively).
- [ ] **#9:** the two `<Component ..._inst>` param-emission blocks in the LEMS
      template are copy-pasted; fold into a Mako `<%def>`.

Phase 2 = native `run(format="brian2")` on the shared core (all_to_all →
`Synapses.connect(condition='i!=j')`; weight outside the saturating gate; oracle =
3-way Deco Brian2-vs-NumPy-vs-LEMS with the RESET-JUMP detector). Retires the
hand-written `deco_column.py`. Phase 3 = NEST or Jaxley proves the core unchanged.

**Prereq before branching the new worktree from dev:** the July 2026 NeuroML fixes
(weight-in-current + `Property name="weight"` jLEMS-load fix, id-collision,
ComponentReference satisfaction, tests) were uncommitted on the
`feat/neuroml-ontology-ingestion` worktree as of 2026-07-21 and are NOT in PR #64.
Commit + merge them to dev first, or the new session builds on a regressed base
(without the Property-weight fix the Deco column will not even load in jLEMS).


## vbjax interoperability backend

Add `vbjax` (Sanz-Leon / Woodman, JAX-based virtual brain library) as a
first-class backend. Goal: every script in
`/Users/leonmartin_bih/work_data/toolboxes/vbjax/examples/` is replicated
by a pure-YAML TVBO `SimulationExperiment` — no Python hacks, no
backend-specific escape hatches in the YAML, no wrapper modules. The
adapter under `tvbo/adapters/vbjax.py` + templates under
`tvbo/templates/vbjax/` are the only places vbjax-specific code lives.

vbjax is structurally close to the existing `tvboptim` JAX backend
(same JAX substrate, same noise-array-as-input idiom, same MPR/JR
neural-mass dfun structure — see
`tvbo/database/models/MontbrioPazoRoxin.yaml` whose `cr`/`cv` parameters
already mirror `mpr_default_theta`). The work is mostly: catalogue the
vbjax primitives, decide for each whether it maps to existing TVBO
schema or needs a small additive extension, then write the
template/adapter pair.

Full design, primitive inventory, per-example YAML replication
strategy, schema additions, and phased roadmap:
**see `dev/Interoperability/vbjax.md`**.

Depends on:
- Backend-in-metadata + per-task dispatch (this file, §2). vbjax
  becomes a `provides_format: [vbjax]` entry on the existing
  `tvbo/database/software/vbjax.yaml` `SoftwarePackage`, with a
  defaults-file entry routing `Integration` tasks to it on demand.
- Backend-independent declarative network construction (this file,
  `NetworkGenerator`). vbjax only needs the **Tier-1** part: one
  named built-in generator (`SphericalGrid`) for the neural-field
  example, plus a `Network.local_connectivity` slot (already a
  first-class concept in classical TVB) backed by `make_shtdiff`.
  The `Procedural` DAG (Tier 2) is Koller2024's concern, not
  vbjax's. Most vbjax examples just load a tractogram or use
  all-to-all coupling — the current `Network` already covers them;
  high-resolution connectome examples (`delays-hcp.py`, `hires.py`)
  only need a `Network.tractogram.format: dense | coo | csr` slot
  so the adapter can pick `make_spmv` vs dense matmul.

Cross-check findings worth surfacing here (full table in
`dev/Interoperability/vbjax.md` §4.1):
- **`bold_dfun` is a Dynamics, not an Observation.** It is a 4-state
  ODE (`s,f,v,q`) with `BOLDTheta` parameters
  (`tau_s,tau_f,tau_o,alpha,te,v0,e0,epsilon,nu_0,r_0` + reciprocals
  + `k1,k2,k3`); `make_bold` integrates it via `heun_step` and then
  samples `v0*(k1*(1-q)+k2*(1-q/v)+k3*(1-v))` as the observation
  read-out. The existing `tvbo/database/observation_models/bold_*.yaml`
  are convolution/HRF-kernel monitors (classical-TVB style) — they
  do **not** carry `s/f/v/q` state or the BOLDTheta fields, so the
  earlier "✅ present as Observation" status was misleading. Action:
  add a new `tvbo/database/models/BalloonWindkessel.yaml` Dynamics
  plus a thin Observation reading the linear sample from its state;
  keep the existing kernel-based bold_*.yaml as a separate family.


## vbi (Virtual Brain Inference) interoperability backend

Add `vbi` (Ziaeemehr, Woodman, Hashemi, Jirsa — INS Marseille,
[`github.com/ins-amu/vbi`](https://github.com/ins-amu/vbi), DOI
[10.5281/zenodo.14795543](https://doi.org/10.5281/zenodo.14795543)) as a
first-class **inference** backend. Goal: every notebook in
`/Users/leonmartin_bih/work_data/toolboxes/vbi/docs/examples/*.ipynb`
(~30 notebooks) reproduced as a pure-YAML TVBO `SimulationExperiment` —
no Python hacks, no escape hatches in the YAML, no per-notebook wrapper
modules. The adapter under `tvbo/adapters/vbi.py` + templates under
`tvbo/templates/vbi/` are the only places vbi-specific code lives.

vbi is **the inference complement** to vbjax. Where vbjax is a JAX
integrator + neural-mass library (one backend, one paradigm), vbi is a
**multi-backend simulator** (Numba JIT, C++ compiled, CuPy GPU, PyTorch
autograd, JAX lazy, TVB kernels) wrapped behind a unified per-model
`run()` API, plus a 40+ time-series **feature library** and two
**posterior-estimation paths** (the `sbi` toolkit — SNPE/SNLE/SNRE with
MAF/NSF/MDN — and a built-in autograd CDE with MDN/MAF estimators in
`vbi/cde.py`). For TVBO this lands three things at once:

1. **`Inference` Task** — the headline SBI workflow (sample prior →
   simulate → extract features → train density estimator → sample
   posterior) becomes a first-class Task class, complementing the
   existing `Optimization` and `Exploration` tasks. Already implied
   by SED-ML §4.2.1.
2. **Feature-extraction pipeline** — vbi's `cfg` dict (40+ features
   organised by domain: statistical, spectral, FC/FCD, complexity,
   event-detection) becomes a structured TVBO `Observation` pipeline.
   Closes the long-standing "Improve Observations" todo (this file).
3. **Backend-per-model routing** — vbi keeps Numba / C++ / CuPy /
   PyTorch / JAX / TVBk implementations of the *same* model under
   `vbi/models/<backend>/<model>.py`. This is the cleanest possible
   stress-test for the Backend-in-Metadata work (this file, §2): the
   YAML carries `task.software: vbi-numba` (or `vbi-cupy`, etc.) and
   the adapter dispatches. Same model + same parameters + different
   backend keys = identical results modulo numerical noise.

Full design, primitive inventory, per-notebook YAML replication
strategy, schema additions, and phased roadmap:
**see `dev/Interoperability/vbi.md`**.

Depends on:
- Backend-in-Metadata + per-task dispatch (this file, §2). vbi
  registers six renderer keys on one `SoftwarePackage`:
  `vbi-numba`, `vbi-cpp`, `vbi-cupy`, `vbi-jax`, `vbi-pytorch`,
  `vbi-tvbk` — exactly the "one package, many `provides_format`"
  case the schema slot was designed for.
- `Inference` Task class from SED-ML §4.2.1
  (`dev/Interoperability/SedML/plan.md`). vbi is its first real
  consumer.
- `Observation` pipelines (this file, "Improve Observations"). vbi's
  `cfg` is essentially a feature-pipeline spec; TVBO needs the same
  structure to express it natively.
- Revisit Heterogeneous parameter specification (this file). vbi's
  `_as_1d_array_like()` broadcasting (scalar → per-node vector → per-sim
  matrix) is exactly the schema gap that todo motivates.
- vbi shares neural-mass models with vbjax and the classical TVBO DB
  (MPR, JR, WilsonCowan, Stuart-Landau, RWW/WW, VEP, BVEP-family).
  Reuse the existing `tvbo/database/models/*.yaml` entries; only
  `DampOscillator`, `GHB`, `RWW` (full Wong-Wang), and `Stuart-Landau`
  need new DB entries.


## Drop `use_ontology` / `_skip_ontology` flags once IRI handling is canonical
- Today there are ~37 occurrences across `tvbo/` of `use_ontology`, `_skip_ontology`, `_populate_from_ontology*` runtime flags that gate ontology backfill.
- Once IRI is the canonical way to declare a sourced component, the flag becomes redundant: **iri present → use ontology/DB data; iri absent → fully self-contained spec.**
- Override semantics should follow YAML/dict merge: ontology defaults are the base, user-provided fields override key by key. Example target:
    ```python
    Dynamics(iri='tvbo:ReducedWongWang', parameters={'a': {'value': 2}})
    # → loads all parameters/state_variables from ontology, then overrides only a.value
    ```
- Cleanup steps:
    1. Remove `use_ontology` / `_skip_ontology` parameters from `DynamicalSystem.__init__`, `Dynamics.from_*`, `Coupling.*` and any other class constructors.
    2. Remove the explicit `_populate_from_ontology_by_name()` / `_populate_from_ontology()` call sites — they become unconditional inside the single `_resolve_iri` step from the previous TODO.
    3. Ensure parameter/state-variable merging is non-destructive: user dict values overwrite at the leaf level (e.g. `parameters.a.value`), not the whole `parameters` slot.
    4. Update tests that pass `use_ontology=True/False` explicitly.


## Experimental / parked

### Composite / coupled Dynamics

A `CompositeDynamics` that bundles N sub-`Dynamics` with declared
**bidirectional** data flow between them, lowered at codegen time to
one combined `dfun` over the stacked state vector. Distinct from the
stateful-Observation pattern (see `Improve Observations` above), which
only handles unidirectional, observer-style coupling without feedback.

Motivating use cases (none of them blocking today):
- Excitatory–inhibitory pairs authored as two separate `Dynamics`
  (Wilson–Cowan, Jansen–Rit decomposition) instead of one merged model.
- Neural–glial / neural–vascular models where the auxiliary
  compartment feeds back into the neural state (unlike BOLD which
  doesn't).
- Multi-compartment single-cell models (soma + dendrite + axon).
- Heterogeneous node groups *within a single node* — e.g. an MPR
  population coupled to a slow-adaptation variable that modulates `η`.
- Possible future home for the vbjax `Dopa` model if/when its
  AMPA / GABA / dopamine sub-states are factored into separate
  `Dynamics` instead of one 6-state monolithic `DopamineQIF.yaml`.

Schema-wise this needs: a `CompositeDynamics` class holding a list of
sub-`Dynamics` plus a `couplings:` block declaring how each
sub-Dynamics' state variables feed into the others' equations. State
naming becomes namespaced (`<sub>.<var>`). Codegen stacks states and
emits one big dfun.

Do **not** materialise until a concrete bidirectional use case shows up
in the roadmap. The vbjax BOLD case explicitly does *not* need this —
it's covered by stateful Observations. Keeping this here so the design
space stays mapped.

## Harmonize `SimulationResult` with tvboptim's `NativeSolution`

`SimulationResult` (tvbo, `tvbo/data/types.py`) and `NativeSolution`
(tvboptim, `tvboptim.experimental.network_dynamics.result`) serve the same
purpose — a single simulation run's output — but differ in interface:

| | `SimulationResult` | `NativeSolution` |
|---|---|---|
| Storage | `xr.DataArray` (named dims + coords) | raw arrays `.data`, `.ys`, `.ts` |
| Time access | `.data.coords["time"]` | `.ts` / `.time` |
| Variable names | `.data.coords["variable"]` | `.variable_names` |
| Repr | `SimulationResult(T, V, N)` | `NativeSolution(shape=..., t=[...], variable_names=(...))` |
| Observations | `.observations` (`Bunch`) | — (returned separately) |

**Goal:** one canonical result type used by all backends.  The JAX backend
already returns `SimulationResult`; tvboptim returns `NativeSolution`.
Options:

1. **Extend `SimulationResult`** — add `.ts`, `.variable_names`, `.ys`
   convenience properties so it is a superset of `NativeSolution`'s
   interface.  tvboptim template wraps its `NativeSolution` in a
   `SimulationResult` before returning.
2. **Absorb into `SimulationResult`** — `SimulationResult` accepts a
   `result=NativeSolution` kwarg (already supported for backward compat)
   and tvboptim's `run_experiment` passes `result=` instead of `data=`.
3. **Deprecate `NativeSolution`** — route it through `SimulationResult`
   everywhere, keep `NativeSolution` as a thin alias for one release.

Option 2/3 is preferred: `SimulationResult` already handles the
`result=NativeSolution` constructor path; the tvboptim template just needs
to use that path and stop exposing raw `NativeSolution` objects to users.

## Thorough analysis of `Integrator.coupling_evaluation` (per_stage vs per_step)
A new backend-neutral integration field (`CouplingStageEvaluation`: `per_stage` /
`per_step`) was added after the Taher2019 replication revealed that tvboptim's Heun
**holds the network coupling constant across the two predictor-corrector stages by
default** (`recompute_coupling_per_stage=False`). For that chaotic + multistable
power grid this selected a *different attractor*: cold-start at K=817 gave 12/438
nodes locked with `per_step` vs **437/438** with `per_stage` (matching an
independent full-Heun reference and the paper). The RHS, coupling sign/weights, and
parameters were all byte-exact — only the intra-step coupling evaluation differed.

This needs a proper study before we trust either default broadly:
- **When does it matter / when is it safe?** Quantify the error vs `dt` (the two
  schemes agree as `dt → 0`, difference is O(dt) in the coupling); characterise
  which regimes diverge (stiff, chaotic, multistable, near-bifurcation, strong
  coupling) vs which are indifferent (linear, weakly-coupled, deep in a single
  basin). Give guidance: "use per_stage when λ₁ ≳ 0 / multistable / matching a
  reference; per_step is fine (and faster) for a single stable working point."
- **Equivalents in original TVB.** TVB's `HeunDeterministic`/`HeunStochastic`
  compute the coupling **once per step** and reuse it in the corrector (coupling is
  passed into `dfun` and not recomputed at the predicted state) — i.e. TVB is
  effectively `per_step`. Confirm this against `tvb.simulator.integrators`, and
  check RK4 and the SDE Heun. If TVB is per_step, the paper's own results (if TVB-
  based) and any TVB cross-validation must account for it. Document the mapping for
  every backend (diffrax = always per_stage; Julia DiffEq = per_stage; ND.jl = ?).
- **Should the default change?** `per_step` is the current tvboptim default and is a
  silent correctness footgun for sensitive systems; `per_stage` is correct-by-
  default but ~2× the coupling-reduction cost per step. Decide per-backend defaults
  and whether tvbo should override to `per_stage` for `method: heun/rk4`.
- **Audit existing results.** Re-run a sample of shipped experiments (esp. chaotic /
  multistable / FIC-tuned working points) under both settings to see which reported
  numbers move; flag any that were implicitly relying on `per_step`.
- **Naming / scope.** Confirm `per_stage`/`per_step` is the right conceptual axis
  (vs a more general "coupling lag" / operator-splitting description), and whether
  the same field should also govern *stimulus/external-input* evaluation across
  stages (currently only coupling).

## Julia backend: mode-axis state layout for multi-mode models

The 4 multi-mode models (`number_of_modes > 1`) — `ReducedSetHindmarshRose`,
`ReducedSetFitzHughNagumo`, `StefanescuJirsa2D`, `StefanescuJirsa3D` — do **not**
run on any Julia backend (DifferentialEquations.jl / NetworkDynamics.jl /
ModelingToolkit.jl). They are currently `xfail`'d in
`tests/functional/test_simulation_backends_julia.py` (`_JULIA_MODE_UNSUPPORTED`),
mirroring `_PYRATES_UNSUPPORTED`.

**Root cause.** Each state variable of a ReducedSet/SJ model is a per-mode
*vector* (length `number_of_modes`, here 3). `mode_dot`/`mode_sum` now render on
Julia (PR #56 → `_mat_dot`/`_reduce_axis` in `codegen/code.py`), and the
array-valued params (`iV0`, `iZV`, `A_ik`, …) make the whole RHS mode-vector-
valued. But the Julia templates lay out state as **flat scalars**:
`tvbo-julia-ODEProblem.jl.mako` builds `u0 = [sv.initial_value for sv in svars]`
(one scalar per state variable) and `tvbo-julia-model.jl.mako` unpacks
`xi, eta, … = x` as scalars, emitting `dx[i] = <rhs>`. So a length-3 mode vector
is written into a scalar `dx[i]` slot →
`MethodError: Cannot convert Vector{Float64} to Float64`
(`setindex!(::Vector{Float64}, ::Vector{Float64}, ::Int)`). Single-mode models
(Epileptor2D/5D, …) are unaffected and pass.

**Fix (real work, deferred).** Give the Julia state a mode axis so each state
variable holds a length-`n_modes` vector: `u0`/`dx` become per-mode vectors
(Vector-of-Vectors, or an `n_svars × n_modes` matrix with row-unpack), the state
unpack in `tvbo-julia-model.jl.mako` yields mode vectors, and the solution
extraction (`solution_to_dataarray` in `tvbo/run/julia.py`, today assuming
scalar state / `n_nodes = 1`) learns the mode dimension. Gate on
`number_of_modes > 1` so single-mode models keep the flat-scalar layout.
Validate against tvb/tvboptim/jax, which already run these faithfully (== TVB to
~1e-16). Then drop the four entries from `_JULIA_MODE_UNSUPPORTED`.

Iterating requires a local Julia runtime (`uv pip install -e '.[julia]'` + the
first-import juliapkg bootstrap + DiffEq precompile); the default dev venv has no
Julia, which is why this was deferred behind the xfail rather than coded blind
against CI.

## Cross-backend divergence on delayed networks (dynamics, not the monitor)

The BOLD observation *monitor* is now consistent across jax/tvboptim/tvb (the
sampling resolver + the causal-convolution fix in
`tvbo/templates/autodiff/jax-function.py.mako`, `mode='full'[:len(x)]`; verified
pointwise-identical on zero-delay networks to ~1e-3). But on *delayed* networks
the backends diverge in the **underlying dynamics**, visible before any HRF
processing (compared via `TemporalAverage` on Lobar8/avgMatrix, RWW):

- `jax` ≈ `tvb` (corr 0.994, mean|Δ| 3e-5), but `tvboptim` **decorrelates** from
  both after the transient (corr ~0.002, mean|Δ| 0.08) despite matching jax to
  1e-3 with delays OFF. So the divergence is delay-specific.
- Time-axis conventions differ: jax `t0≈0.5ms`, tvb `t0≈134.5ms` (history/offset).
- Root cause is per-backend **delay discretization + history initialization**;
  for a nonlinear delayed system this amplifies (sensitive dependence), so
  pointwise identity may be unachievable unless the delay rounding is bit-identical.

**Shipped into 0.5.0 as a silent known-limitation** — delayed-network runs give
backend-dependent results. Open questions: (a) is pointwise cross-backend
identity under delays a goal, or is statistical agreement (FC/spectra) the bar?
(b) unify the time-axis/`t0` convention; (c) investigate why `tvboptim`'s delay
path diverges from both `jax` and `tvb`.

Related, resolved: under delays TVB's native Bold monitor emits **one BOLD
sample past the sim end** (t=10080ms > 10000ms duration; `floor(dur/TR)=13`,
tvb=14). Triple-verified as caused by the delay horizon (max idelay 671 steps ≈
67ms; delays OFF → tvb=13). Accepted as TVB-native boundary behaviour; the
consistency tests neutralise it (zero-delay network + `_align`).

## `ExecutionConfig.compile` / jit toggle (backend-independent)

Make jit/compilation an opt-out, resolved in the adapter, backend-independent.
Do NOT add a jax-specific `jit` field — reuse the existing `compile` *intent*
(today on `Dynamics`, schema ~L1847, `ifabsent: false`, "compile to machine code
where the backend supports it; inherently-compiled backends no-op"). Each backend
maps the intent: jax/tvboptim → `@jax.jit`, numba → `@njit`, MTK → `mtkcompile`,
julia → no-op.

Reconcile the **default-direction mismatch**: `Dynamics.compile` is opt-in accel
(default false — numpy→njit), but jax jit is the *default* execution mode (you
disable only to debug). Fix by making it a **tristate `Optional[bool]`**: unset =
backend's natural default (jax jitted, numpy interpreted), `true` = force
compile, `false` = force interpreted/eager. Home = **`ExecutionConfig`** (an
execution concern, alongside `precision`/`accelerator`), NOT `Integration`
(that's the numerical scheme). Add `run(compile=…)` (jax alias `jit=`) as a
non-persisted runtime override. Template only emits the resolved decision (gate
the `@jax.jit` decorator; no arithmetic in Mako).

Decision still open: **A** = `ExecutionConfig.compile` default + optional
per-model `Dynamics.compile` override (needs precedence rules); **B** = single
`ExecutionConfig.compile`, leave `Dynamics.compile` as its existing narrow hint
(recommended — one name, one meaning). CAVEAT for docs: disabling jit is a
**debugging** aid (concrete values/print/pdb inside dynamics), NOT a single-run
speedup — eager `lax.scan` is slower than paying the compile.

## jit performance defaults (single-run latency)

`run("tvboptim")`/`run("jax")` jit by default, so a one-off run pays full XLA
compile for a single execution (amortised only over sweeps/optimization). Two
cheap wins: (1) enable jax **persistent compilation cache**
(`jax_compilation_cache_dir`) so the first-run compile is one-time across
*sessions*, not per-run; (2) make the default backend **workload-aware** —
single/interactive run → a non-compiling backend (tvb/numpy), sweep/optimization
→ jax/tvboptim. Disabling jit is NOT the fix (eager scan is slower); backend
choice + compile cache are.



## PDF to markdown / KG

- [ ] compare https://github.com/opendataloader-project/opendataloader-pdf with dots-mocr setup (/Users/leonmartin_bih/tools/dots-mocr).

- [ ] explore https://github.com/EPFLiGHT/mmore



## Check SONATA Data format
https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md

- can we adopt?
- can we improve move to more modern state-of the art?
- Is it BIDS compatible?



## Engine-agnostic `workflow.setup` (dedup env activation for multi-engine studies)

`WorkflowEngineConfig.setup` (verbatim shell lines run before the workload, e.g.
`conda activate <env>`) is declared **per engine** — `workflow.slurm.setup`,
`workflow.snakemake.setup`, `workflow.nextflow.setup` — like `env`/`venv`/`modules`.
A study run under more than one engine must repeat the same activation lines in each
block, but env setup is engine-independent, so that's redundant.

**Plan.** Add an engine-agnostic `workflow.setup` (on `WorkflowConfig`, sibling of
the per-engine blocks) that every emitter prepends to its own
`workflow.<engine>.setup`. Precedence: shared `workflow.setup` first, then per-engine
`setup` (an engine can add to / override after the shared lines). Touches: schema
`WorkflowConfig.setup` + `make gen-linkml`; `merge_workflow_spec` / `plan()` to fold
the shared list into each `engine_block['setup']` (reuse `_as_lines`); templates
already consume `sb.get('setup')` unchanged. Keep per-engine `setup` for
engine-specific lines.

Context: per-engine `setup` hook + `tvbo workflow submit` landed 2026-07-13. The
per-engine model is consistent with `env`/`venv`/`modules`; this only removes the
multi-engine duplication.

## tvboptim: sparse coupling for delayed / nonlinear per-edge `pre` (framework gap)

Instantaneous *vectorized* (source-only) sparse coupling is DONE tvbo-side: the factored
angle-addition form (`pre=[sin,cos]`, `post` recombine) + `Network.graph_representation:
sparse` emits `SparseGraph` in the experiment codegen (Taher2019 sweeps: ~7.7× faster,
byte-identical, tested in `tests/test_codegen_sparse_graph.py`). What remains is a genuine
**tvboptim** primitive gap for the two cases that must stay per-edge:

- a **delayed** difference coupling — θ_j(t−τ_ij) is edge-indexed, so it can't reduce to a
  per-node mat-vec; and
- any **nonlinear per-edge `pre`** (sigmoidal Jansen-Rit, hyperbolic tangent, …).

tvboptim's sparse per-edge path (`experimental/network_dynamics/coupling/base.py`,
`_compute_sparse_incoming` → `_sparse_pre` → `jsparse.sparsify(self.pre)`) requires `pre`
to be sparsity-preserving, so a nonlinear `pre` (e.g. `cos`) raises
`NotImplementedError: sparse rule for cos … would result in dense output`.

**Fix (small, local, general):** in the sparse per-edge path, replace
`jsparse.sparsify(pre)` with an element-wise-over-`nnz` evaluation — gather source/target
states at the edge indices (the dense edge-gather already exists in
`_sparse_weighted_sum(..., is_bcoo=False)`), apply `pre` element-wise, then scatter-sum
`pre_vals * weights.data`. This unblocks *every* nonlinear coupling on sparse graphs; tvbo
needs nothing beyond emitting the factored `pre`/`post` (already supported).

Not on the critical path for Taher — its delayed control runs are single sims (cheap,
local) and correctly fall back to dense via the `use_sparse` gate (instantaneous +
vectorized only). Surfaced 2026-07-15 while adding the sparse codegen.

## Progress logging for exploration / sweep `lax.scan` (JAX-native `io_callback`)

PR#63 (central logging) wired `LoggingProgressCallback` into the OPTIMIZER path only
(optax loops, `tvbo/templates/tvboptim/callbacks.py`). The EXPLORATION / sweep path —
`adiabatic_scan` (hysteresis + operating point), `lyapunov_branch` (Benettin), and the
vmapped exploration grid — runs as one JIT-compiled `jax.lax.scan`, so it logs
`STEP 2 > <exploration>` and then nothing until it returns. For a long sweep (Taher exp 30
was ~8 h before the sparse win) the .out stays empty the whole run — the cluster
"empty log" complaint.

**Fix (JAX-native, no JIT break):** `jax.experimental.io_callback` (already in the env)
fires a host callback from *inside* a `lax.scan`. Carry an iteration counter in the scan
carry and call `io_callback(log_progress, ..., k, total)` every N steps → stream
"K 42/181 …" through the central `tvbo.run` logger, gated by `TVBO_LOG_LEVEL`. This is the
`jax_tqdm` pattern (~15 lines; jax_tqdm itself need not be a dependency). Wire it into the
sweep / Lyapunov / grid codegen partials (`tvbo-tvboptim-sweep.py.mako`, the exploration
`<%def>`s). Surfaced 2026-07-15 running the Taher Lyapunov + hysteresis sweeps (they went
silent after `> lyapunov_branch` / `> hysteresis_sweep` despite finishing in ~15–25 min).

## Figure spec — declarative figures from result containers

Close the last open end of full study replication: **map result data → publication
figures** declaratively. Design agreed + schema stub drafted 2026-07-16 — full write-up
in `dev/figure-spec-design.md`.

LinkML-native `Figure/Panel/Layer/DataRef` (+ Encoding/Guard/ColorbarSpec/Style) in `schema/`
(stub: `schema/figure.yaml`, /simplify-reviewed to 8 classes reusing tvbo's name/description/
label/iri + Provenance/Argument; wired into tvbo_datamodel + regen; SimulationStudy.figures
added). Rendering is **codegen** (not a runtime interpreter): a bsplot adapter
(`tvbo/adapters/bsplot.py`) resolves context + a Mako tree (`tvbo/templates/bsplot/`) emits a
self-contained `plot.py` — same machinery as sims/reports; the emitted script IS the
replication deliverable. `figure.render_code('bsplot')`/`.render('bsplot')` mirror
`experiment.render_code`/`.run`. 2nd backend = new template tree, not a rewrite. Turns each
study's hand-written `plot.py` into a generated one + a registered escape hatch (`custom`
panel callable / `transform`) for the bespoke.

Resolved: unified `Panel` with `kind` enum (`cartesian`/`heatmap` grammar + `custom`/`image`
peers; `surface`/`network` reserved); data binding `used: {experiment: IRI, output, sel}`
(= PROV `used`, label-keyed); compute ladder Observation → declarative-postproc → callable;
**per-panel** backend (mixed-backend figures prepared, MVP all mpl); lightweight
PROV-by-`slot_uri`; **insets/colorbar/legend** first-class (per-node mini-plots, marginals,
zoom, pinned inset colorbars).

Build (walking skeleton): (1) ✅ wire `schema/figure.yaml` in + LinkML regen + SimulationStudy.figures;
(2) codegen — `tvbo/adapters/bsplot.py` (resolve context) + `tvbo/templates/bsplot/` Mako tree
(`<%def>` partials `panel_cartesian`/`panel_heatmap`/`panel_image`/`panel_custom`, mosaic via
`bsplot.figure.subplots`, `style.use`, `format_fig`, guard→placeholder, insets) emitting `plot.py`;
(3) IRI→container binding (skip `*_network.h5`, `observation__`, `sel`, sidecar params);
(4) **PROOF**: `render_code('bsplot')` a Jansen1995 figure, run vs `output/nc/*`, diff the original;
(5) widen to Taher2019 (bifurcation/sweep/A-B/placeholder/insets); (6) registries + `tvbo figure render`;
(7) **workflow/HPC**: figure-rule `<%def>` in `tvbo/templates/workflow/{snakemake,slurm,nextflow}/` — inputs =
the figure's `used` containers (provenance graph = DAG), resources = `Figure.workflow_overrides` (reuses
`WorkflowConfig`, added) over study `workflow`; `tvbo workflow submit/snakemake <study>` emits figure rules
alongside experiment rules so heavy renders run as their own SLURM jobs after their experiments. Emitted
`plot.py` stays user-editable.

**BUILT (2026-07): walking skeleton + fidelity + workflow emitter.** `tvbo/adapters/bsplot.py`
(resolve context, `TRANSFORMS` up/down/order_by_branch, `CUSTOM_PANELS` registry, `sel`, axopts) +
`tvbo/templates/bsplot/*.mako` (emits self-contained `plot.py`; cartesian/heatmap/image/custom; guard→placeholder).
`tvbo/adapters/figure_workflow.py` + `tvbo/templates/workflow/snakemake/` emit figure Snakemake rules
(`used`→`input:`, `workflow_overrides`→`resources:`). **Taher2019 Fig 5 reproduced — 8 panels, plotted data
bit-identical to the hand-written `_sweep_figure`**; Jansen1995 image montage. Rendering contract in the schema:
`Figure.{width,height (mm), dpi, font_size (pt, real), auto_format, panel_numbers, panel_number_format,
panel_number_loc}` + `Panel.annotations` (Annotation class) + legend via `Panel.opts.legend`. Pipeline HTML schema at `docs/Replication/pipeline.qmd`.

**MVP FINALIZED (2026-07).** CLI `tvbo figure render <spec>` (`tvbo/cli/figures.py`) loads a Figure/Study YAML and
renders each figure; `tvbo workflow snakemake <study>` now appends figure rules (`figures.smk` + `plot_<name>.py`) when
`study.figures` present. Rendering contract fields (physical mm/pt sizing, real font, dpi, auto_format toggle,
panel_numbers + format + loc), `Panel.annotations` (Annotation class), study `.mplstyle` support (`plt.style.use` for
paths, `bsplot.style.use` for names — bsplot can't take paths). **17 figure-codegen tests** (`tests/test_figure_codegen.py`)
+ datamodel green. PROOF-OF-CONCEPT: Taher2019 **Fig 5 reproduced declaratively, data bit-identical**, closely matching the
paper (structure, LaTeX labels, (a)-(h), circled K) — `dev/figure-demo/ab_fig5_final.png`.

By design, NOT gaps: cross-container x/y merge + data-derived shared ranges live in `custom` callables (the escape hatch —
grammar can't express them). Out of scope: λ₁ magnitude paper-vs-run gap (a sim/analysis issue, pre-existing).
PIXEL-PARITY POLISH DONE: legends conditional (paper has none), `Figure.height_ratios`/`width_ratios` layout control added,
panel-label collision fixed (K≈ moved to the opposite corner from the panel letter), study `.mplstyle` via `plt.style.use`.
`dev/figure-demo/{fig5_final,ab_fig5_final}.png` — Taher Fig 5 declaratively, closely matches the paper. 277 tests green.
REMAINING (minor/deferred): λ₁ magnitude sim gap (not rendering); a negligible ylabel clip on the tall top panel;
slurm/nextflow figure emitters (snakemake primary); a `Figure.render()` method (funcs + CLI already cover it); kit
`OUT_DIR` vs local `output/nc/` path reconciliation.

## Investigate 3× post-rate magnitude in HeterogeneousEdges (most likely our pyrates codegen)

`docs/Networks/HeterogeneousEdges.qmd` (pre → {Depression, Facilitation,
Tsodyks–Markram} synapses → post; each synapse→post edge `weight=0.33`, `source_var:
r_eff`) renders a post-synaptic rate peaking ~2.4. That is **~3× too high**.

**Evidence (2026-07-24).** Rebuilding the identical 5-node network on the tvboptim
heterogeneous engine — where `r_eff = r_in·x·u` is routed correctly via an
input-dependent readout `readout(state, params, inputs)` — gives post peak **0.79**.
Forcing the synapse→post weights to **1.0** in tvboptim reproduces pyrates exactly
(**2.40 ≈ 2.398**, ratio 3.0×). So the tvbo→pyrates path effectively **drops the
0.33 edge weight** on edges whose `source_var` is an output/derived variable
(`r_eff`). The *rising trend* is correct (the Facilitation pathway `r_in·u`, `u`
climbing, dominates the sum) — only the magnitude is wrong.

**Prime suspect: our codegen, not pyrates.** tvbo generates the pyrates model, so a
dropped edge weight most likely comes from `tvbo/codegen/pyrates.py` /
`tvbo/adapters/pyrates.py` not applying the edge weight when the source is a
`derived_variable`/`output`. `r_eff` depends on the coupling input `r_in` — the
output-depends-on-input case pyrates handles awkwardly (the doc already notes
pyrates "cannot monitor output variables"). Do NOT assume tvboptim is the
reference: the use-case implementation (equations, `r_eff` formulation, weights)
may itself be off.

**Actions.**
- Trace an `r_eff`-source edge through `to_yaml(format="pyrates")` / the pyrates
  codegen: is the `0.33` weight emitted on the pyrates edge, or lost for
  derived-variable sources? Control: check a state-var source (`r_out`) edge.
- Validate the 3-synapse dynamics AND magnitude against the **original
  Tsodyks–Markram study** and the PyRates short-term-plasticity tutorial this
  example is based on, to decide the correct post magnitude before blaming an
  engine.
- If it is a pyrates-side bug (not our codegen), confirm and report upstream.
- Once resolved, regenerate the `HeterogeneousEdges` figure (via the tvbo→tvboptim
  heterogeneous adapter once P1/P2 land, or after fixing the pyrates codegen).
- Repro: 5-node pre→{dep,fac,tso}→post, drive pulses amp 5 / width 15 at
  [50,80,110,140,170, 500,…]; compare `exp.run("pyrates")` vs the tvboptim
  `HeterogeneousNetwork` rebuild (both reconstructed during this investigation).

**Finding (2026-07-24) — the use-case TM equations are non-canonical (2nd, independent
bug).** Checked against Cortes2013's "Standard TM Model" (deterministic continuous
TM): `ẋ = (1-x)/τ_D − u·x·E`, `u̇ = U·E·(1-u) − (u-U)/τ_F`, drive ∝ `u·x·E`. The
doc's synapse matches on `r_eff = r_in·x·u` and the facilitation decay, but invents
two coefficients the real TM does not have:
- depression `k·x·u·r_in` with **k=0.5** — should be coefficient **1** (`u·x·r_in`);
- facilitation increment `k_fac·(1-u)·r_in` with **k_fac=0.05** — should be
  `U0·(1-u)·r_in` (the increment coefficient IS the baseline `U0=0.2`, not a free
  `k_fac`).
Inherited from the PyRates tutorial the doc cites, not from Tsodyks–Markram. So two
independent bugs: (1) this model-form deviation and (2) the codegen weight drop
above. Fix the equations to `-u·x·r_in` and `+U0·(1-u)·r_in`, and wire a citation
(`Tsodyks1997` spike-based; `Cortes2013` deterministic) into `HeterogeneousEdges.qmd`.
Paper extraction: `tvbo-manuscript/use-cases/replication_studies/Cortes2013/original_study/`.

## `jax` backend silently ignores `explorations` — warn, then support (2026-07-24)

Discovered while building the Cortes2013 replication: running a `SimulationExperiment`
that declares an `Exploration` (a `space` sweep or `n_trials` ensemble) on the **`jax`**
backend runs a **single** forward sim and drops the sweep entirely — `result.explorations`
is empty and `result.save(...)` writes `[]` (nothing). No error, no warning; it just
silently ignores the exploration. Only **`tvboptim`** currently executes explorations
(the vectorised grid path). This cost real debugging time (the run "succeeds" and writes
nothing).

- **Step 1 (guard).** When an experiment carries a non-empty `explorations` (or
  `optimizations`/`inferences`) and the selected backend is `jax` (or any backend without
  exploration support), **raise or warn loudly** at dispatch — e.g. in
  `tvbo/cli/run.py::_effective_backend` / the backend's `run()` entry — telling the user
  the sweep will not run and to use `tvboptim`. Silent no-op → empty save is the worst case.
- **Step 2 (support).** Add jax-native exploration execution so a plain forward sweep does
  not require the full `tvboptim` optimisation stack — vectorise the grid (`vmap`/`lax.map`)
  and populate `result.explorations` the same way tvboptim does, so `jax` and `tvboptim`
  produce interchangeable result containers for a bare sweep.

## Backend-independent schema representation of integration schemes (2026-07-24)

Surfaced twice during the Cortes2013 replication:
1. The tvboptim **heterogeneous** adapter path maps the solver by
   `str(method).lower()` against `{"euler","heun","rk4","rungekutta4"}` and silently
   **falls back to Heun** for anything else — so `method: RungeKutta4thOrder` (a valid
   name elsewhere) becomes Heun there (`tvbo/adapters/tvboptim.py:540-546`). An explicit
   scheme at too-large `dt` then sustains lightly-damped librations (wrong attractor).
2. The tvboptim **codegen** template uses a *different* SOLVER_MAP
   (`['euler','heun','heunstochastic','rk4','rungekutta4thorder','runge_kutta',
   'rungekutta']`) — so the two paths accept different spellings of the same scheme, and
   `RungeKutta4` (no `thorder`) errors in one and Heun-falls-back in the other.

This name-matching is flaky and duplicated. **We need an abstract, backend-independent
metadata representation of integration schemes** (tvbo already has the seed of this):
- a **database of simple solvers as schema** (Euler, Heun, RK4, …) with their Butcher
  tableau / order / stability metadata, so every backend resolves the SAME canonical
  scheme identity instead of string-matching per-adapter;
- **complex/backend-specific solvers** (e.g. `Rodas5`, `Tsit5`, adaptive DDE solvers)
  live in the database too but are **linked as callables** scoped to the language/backend
  that provides them;
- adapters consume the resolved scheme metadata (canonical id + per-backend callable),
  never a raw lowercased name — killing the silent Heun fallback and the spelling drift.

Same spirit as the "backend-independent metadata states INTENT, not one backend's
mechanism" rule. Until then, name the solver exactly as each backend's map expects and
verify the integrator against a dt-converged reference (Phase 7) near sensitive regimes.

## tvboptim exploration path ≠ single-sim path at the same IC (2026-07-24)

Diagnosed in the Cortes2013 Fig-4 (near-homoclinic transient chaos). At a **nominally
identical** initial condition, the tvboptim **exploration/`n_trials` grid path** and the
tvboptim **single-sim path** produce trajectories that differ by ~7e-5 at the first recorded
step and grow apart. Established:
- It is **not** the integrator scheme: rk4 / heun / euler all give the *same* exploration
  result (all ride the orbit; single-sim + NumPy reference settle).
- It is **not** precision: both paths are float64.
- A **single** tvboptim sim reproduces the dt-converged NumPy RK4 reference (both settle,
  peak 17.40, diff ~1e-3). Stable limit cycles match the reference *exactly* — so the paths
  agree everywhere the dynamics is non-chaotic.
- Near the Shilnikov homoclinic the system is genuinely chaotic (σ≈16, λ₁≈17 s⁻¹, e-folding
  ~0.06 s), so the ~1e-4 path discrepancy amplifies over 15 s into a qualitatively different
  transient (settle-to-down vs ride-the-orbit). This part is *expected physics* (the paper's
  sensitive-dependence result), not a defect.

**Root cause = fractal basin + a confounded comparison; NOT a codegen or sampler bug.**
Established:
- A **deterministic exploration** (`space` sweep, `transient_time=0`, ICs via `initial_value`,
  NO `distribution`) reproduces the single-sim path **bit-consistently** at the same IC
  (`E0_rec=2.97558`, 2 spikes, settle, peak 17.40 = plain sim = NumPy reference). Grid/vmap
  codegen, RHS, and settings/IC passing are all correct.
- The `_sample_initial_conditions` codegen (`tvbo/templates/tvboptim/
  tvbo-tvboptim-experiment.py.mako:1574-1596`) replaces each trial's state row with
  `jax.random.uniform(key, (n_nodes,), minval=lo, maxval=hi)` — the right behaviour for an IC
  ensemble; a degenerate `{lo:X, hi:X}` returns `X`.
- The apparent "same IC → different outcome" was a **confound in the diagnostic**: the
  ensemble's *recorded* first sample is taken after the first integration step (≈E(dt)), not
  the true sampled IC, so "single sims at the ensemble's E(0)" actually started ~1e-4 (one
  step) off the true ICs. Near the Shilnikov homoclinic (σ≈16, e-folding ~0.06 s) the basin is
  fractal, so ~1e-4 flips settle-vs-orbit. Genuine sensitive dependence (the paper's result).

So there is **no confirmed bug**. Recommended *guardrails* (not a fix for a known defect):
(1) add a determinism test asserting `grid-path == single-path == reference` bit-for-bit at a
fixed IC in a non-chaotic regime (and a degenerate distribution returns its point exactly);
(2) consider recording the true IC (E at t=0, pre-step) so a trial's sampled IC is inspectable
without the one-step offset. Until then, integrate sensitive-regime transients on the
single-sim path (or the dt-converged reference) — Phase 7.

## Streaming monitor reducer: single-state deferred readout breaks the scan carry — ✅ FIXED 2026-08-04

`tvbo/templates/tvboptim/tvbo-tvboptim-observation.py.mako`, monitor-reducer
`output_per_step=False` branch. For an observer with **exactly one** state variable codegen
emitted `return (_new_x), None` — parens around one expr are **not a tuple** — and unpacked
`x = carry`, while `_init`/`tuple(_st)` supply a 1-tuple → `jax.lax.scan` carry mismatch at
trace time. BOLD (4 states) hid it.
**Fixed:** trailing commas so a 1-element carry stays a tuple — `…snames…, = carry` (both the
`_step` and `_chunk` unpacks) and `return (…_new_x…,), None`. Verified with a single-state
`output_per_step=False` repro (`/tmp/gen34_harness/repro_single_state_monitor.py`) + 27/27 suite
still green. STILL TODO: add that single-state case as a permanent regression test (none exists
— why 27/27 passed before). (code-review 2026-08-04, #5.)

## Streaming monitor reducer: guard forbids the host whole-trajectory block it can receive (2026-08-04)

Same file, monitor `_update`: `if _rem and _n_full: raise` rejects a block longer than one
period but not a whole number of periods. This is a **deliberate, tested** contract
(`tests/test_streaming_monitor_reduction.py::test_a_block_that_is_not_a_whole_number_of_periods_raises`;
streaming blocks are period-aligned by `streaming_post_eval_plan`). BUT `AbstractMonitor.__call__`
folds the **entire trajectory as one block**, and `_init` (`n_steps // _period` slots) + the
tail branch already *handle* a remainder — so a **host-evaluated monitor whose run length
isn't a multiple of period-in-steps crashes** (e.g. BOLD TR=720ms, dt=0.1 → 7200 steps;
n_steps=1e6 → raise). The 27 tests pass only because they use period-multiple lengths.
**Decision needed — do NOT just delete the guard (it breaks the test):** either (a) require
TR-multiple run lengths upstream and keep the strict guard, or (b) let the reducer drop the
partial tail on the host path (remove the guard, update the raise-test). (code-review 2026-08-04, #1.)

## Whole-brain post-tuning evaluation must stream — exp_34 hung (2026-08-04)

exp_34 (`~/work/schirner2023replication/schirner34_validate`, SLURM 28169921) **hung in
post-tuning** and was **cancelled by the user at 11:43** (no result — `results/34/` empty).
Timeline: `fic_eib complete!` at 09:58, entered post-tuning (2nd warmup, 300 samples), then
~1h45m of zero output. The tuning itself converged (mean_H_e ≈ 4 Hz). The hang is the known
whole-brain failure mode — the **post-tuning evaluation materializes a full-length 379-node
trajectory** unless the post-eval streams, blowing memory → stall.
**Corrected root cause (2026-08-04):** NOT a missing stream — the `schirner34_validate` kit
*already* declares BOLD `reduce: streaming`, so memory was fine. The real bottleneck is
**compute**: `integration.duration: 36e6 ms` @ `step_size 1 ms` = **36M steps** over 379 nodes,
and `accelerator: auto` → the CPU node `hpc-cpu-96`. The post-tuning eval re-integrates that whole
sim on CPU (no progress ticker) → looks hung. **Fix = GPU** (user chose GPU, full duration).
**DONE:** emitted `exp34gpu_out.tar.gz` (`spec_gpu34/`, `accelerator: gpu`, `partition: gpu`,
dropped `JAX_PLATFORMS=cpu`; koller `.venv` is jax[cuda]); ran as SLURM 28212657 on hpc-gpu-2.
**OUTCOME (2026-08-04, measured):** the GPU run is HEALTHY but the GPU barely helps — `nvidia-smi`
= 54% util / 2.5 GB VRAM, `sstat` = one CPU core pinned 100% / 2.4 GB RSS. It's a **launch-latency-
bound sequential scan** (36M steps, 379 nodes) — NOT OOM (streaming already bounds RAM), NOT
deadlocked, just dispatch-bound-slow. So a single-subject sequential run doesn't benefit from a
GPU; letting it finish for the result, but the RIGHT default is **single subject → CPU, batched
cohort → GPU** (the on-device cohort fills the GPU). See [[reference-single-subject-gpu-launch-bound]].
**Follow-ups below.**

## exp_34 post-tuning finalization: ~45 min of silent CPU-bound compute after the BOLD eval (2026-08-04)

After the post-tuning BOLD ticker hit `50000/50000`, the job spent 45+ min GPU/CPU-busy with NO
log output and no result yet — a phase (FC compute + observations + result assembly/save over
50000 TRs × 379) that emits no `i/N` ticker. Not a hang (it's computing), but it's an opaque
black box. **Improve:** add per-phase elapsed + a completion log to the post-tuning evaluation /
finalization in `tvbo-tvboptim-experiment.py.mako` (STEP 3 post-eval), and confirm the finalization
isn't doing redundant work (e.g. a second full-duration sim, or an O(T²) FC step). The new `_log`
`[+Xs]` prefix + per-algo `complete! (tuning Xs)` (done this session) cover tuning; extend the same
to the post-eval so a busy-but-silent phase is never mistaken for a hang again.

## Accelerator default guidance: single→CPU, cohort→GPU (2026-08-04)

Measured that a single-subject sequential fic_eib run is launch-bound on a V100 (above). Consider a
gentle heuristic/log-note when `accelerator: gpu` is requested for a single-subject sequential
experiment (no cohort vmap, no wide exploration fan) — "GPU may not help a sequential run; consider
CPU or an on-device cohort." Keep it advisory (backend-independent intent), not a hard block. The
benchmark TSV + phase timeline now make the CPU-vs-GPU-vs-cohort call **data-driven** — re-benchmark
before committing. Ties into the tune-on-CPU / eval-on-GPU `from_experiment` split idea above.

## Commit the uncommitted streaming + initial_value fixes (2026-08-04)

Working tree (dev) has, uncommitted and mixed: the **initial_value refactor** crash-fixes
applied 2026-08-04 (missing `initial_value` imports in `gillespie.py`/`neuroml.py`/`run/graph.py`,
pyrates-template indentation, `julia_model.py` migration) + the **streaming #3/#7** fixes
(`observation.py.mako` period `float(to_numeric)`, collapsed comment blocks) + pre-existing
streaming/initial_value WIP. Cohort work is already committed. Split into logical commits (one
initial_value crash-fix, one streaming reducer); `git add -p` unavailable in-session, stage by file.

## on_device cohort: per-subject results carry only `estimate__*` (2026-08-04) — by design, revisit

An on_device cohort saves per-subject `sub-<id>_..._result.h5` with only the tuned params
(`estimate__J_i/wLRE/wFFI`), NOT per-subject observations/optimizations (on-device tunes
parameters, not per-subject trajectories). A study whose figures read
`observation__*`/`optimization__*__final_loss` works under `fan_out` but silently `KeyError`s
under `on_device`. Deliberate (mirrors the fan-out param contract) — but consider a warning at
save time, or a post-tuning per-subject eval when a downstream figure needs per-subject
observations. (code-review, left deliberately.)

## on_device large cohort writes N duplicate connectome sidecars (2026-08-04) — efficiency, revisit

`_save_per_subject` calls the full `ExperimentResult.save` per subject, so `freeze_yaml`
re-writes an identical `sub-<id>_..._network.h5` for every subject (N copies of the shared
connectome). Mirrors fan_out (each fan_out job also freezes the shared net), so per-subject
results stay self-contained — but a 200-subject whole-brain cohort writes 200 identical dense
connectomes. Consider a shared-network reference for big cohorts if disk/serialization bites.
(code-review, left deliberately.)

## on_device auto batch_size double-compiles the fit (2026-08-04) — minor

`resolve_cohort_batch_size` auto path (`dataset.batch_size` absent) AOT-compiles the full
single-subject fit via `estimate_per_cell_bytes` just to size the batch, then the run compiles
the vmapped fit — two full compiles of an expensive fit. Mirrors `n_parallel:auto` (consistent),
only on the auto path, avoidable with an explicit `batch_size`. Optimize with a cheaper peak
estimate (e.g. reduced iterations) if compile time bites. (code-review, accepted.)
