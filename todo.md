# TODO

## Migrate runtime ontology: deprecated `tvb-o.owl` → generated `tvbo.owl`

Switch every runtime consumer from the deprecated **class-based**
`tvbo/data/ontology/tvb-o.owl` (1516 classes / 173 individuals; `JansenRit`
is an `owl:Class`) to the generated **individual-based** `ontology/tvbo.owl`
(422 classes / 1236 individuals; `JansenRit` is an `owl:NamedIndividual` with
explicit `tvbo:hasParameter`/`hasDerivedVariable` edges), built by
`make gen-merged`. **Preserve** the deprecated file — do not overwrite it; keep
it as a frozen parity reference.

Single load point is `tvbo/ontology/owl.py:131-149` (everything reuses
`owl.onto`). Phase A (platform, low risk): package the generated owl under
`tvbo/data/ontology/tvbo.owl`, repoint the loader, and teach
`DirectOntologyAPI.get_class_hierarchy()` + `query.py` to include
`owl:NamedIndividual`. Phase B (larger): rewrite `owl.py`'s class-based
high-level API (`get_models`/`get_model_parameters`/… → `.instances()` +
`hasParameter`/… object properties). **With `owl.py`** and the full
file-by-file change list, impact map, and verification: **see
`dev/runtime_ontology_migration.md`**.

## Harmonize class names with `tvboptim`

Rename `ExplorationAxis` → `Axis` and reshape it so tvbo can declaratively
specify any `Space` configuration that tvboptim supports
(`GridAxis`, `LogGridAxis`, `UniformAxis`, `DataAxis`, `NumPyroAxis`).
`Space`/`ExplorationSpace` become aliases of `Exploration`; the slot
`Exploration.space` becomes `Exploration.axes`.

Full design, rationale, file-by-file impact, and step-by-step
implementation plan: **see `dev/tvboptim_harmonization.md`**.

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
