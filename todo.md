# TODO

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


## Improve Observations

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
