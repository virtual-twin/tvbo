## 1.0.0 – 2026-08-24

### Added
- **`tvbo.__all__` and `__dir__`.** `dir(tvbo)` now lists the API rather than the
  stdlib modules (`os`, `shutil`, `tempfile`, `warnings`, `logging`) that `import
  tvbo` bound as a side effect of its own setup. `__all__` is what SemVer covers;
  see the new *What is public* section of `CONTRIBUTING.md`.
- **`tvbo/py.typed`.** The package is annotated throughout, but without PEP 561's
  marker file every downstream `mypy` ignored those annotations.
- `SECURITY.md`, issue templates and a PR template.
- Schema-declared `aliases:` now resolve at load time, so `dt` (for
  `Integrator.step_size`), `number_of_regions`, `righthandside`/`lefthandside` and
  `components` are accepted where the schema says they are. Resolution happens per
  class, so a free-form key — a model parameter named `dt`, a `components:` list
  under an unrelated block — is never rewritten.
- Experiment YAMLs support `!include` and merge keys (`<<: *anchor`), which
  networks and studies already had.

### Changed
- **`PDESolver` is a `Solver`.** It inherits `method`, `abs_tol`, `rel_tol` and
  `step_size`; the duplicate `time_integrator` and the untyped string `tolerances`
  are gone. Migrate `time_integrator:` to `method:` and `tolerances:` to
  `abs_tol:`/`rel_tol:`; `dt:` still works as the alias of `step_size`.
- One reader resolves the noise amplitude for every backend (tvboptim, Brian2,
  NetworkDynamics.jl, Julia). Declare it as `parameters: {sigma: ...}` (a standard
  deviation) or `parameters: {nsig: ...}` (a dispersion, sigma = sqrt(2*nsig)).
- The openMINDS export walks `is_a`, so a subclass carries its inherited slots
  (`PDESolver` and `Integrator` had been exported without theirs).
- `tvbo workflow submit --cores N` overrides only the profile's *executor*. The
  kit still runs in the container its profile declares, instead of dropping the
  profile — and its bind mounts, retries and keep-going — entirely.

### Removed
**1.0 clears the deprecation backlog rather than carrying it.** Every name that
warned in 0.5.x is gone; each had a replacement that was already the documented
path, and the callers in `tvbo/`, `tests/` and `docs/` are migrated to it.

- `noise: {intensity: ...}` — the schema slot is removed, so a recipe that still
  declares it is rejected instead of silently read as a standard deviation. Use
  `parameters: {sigma: ...}`, or `parameters: {nsig: ...}` if the value was a
  dispersion (the two differ by sqrt(2D)/D). One reader, `utils.noise_sigma`,
  now knows exactly two spellings.
- `random=True` / `random_initial_conditions=True` on `Dynamics.get_initial_values`,
  `SimulationExperiment.run` and `collect_initial_conditions` — declaring a
  `distribution` on the state variable is the only way to ask for sampling. The
  flag sampled every variable's raw domain regardless of what the model declared.
  `plot.dynamics` spread its multi-trial starts through that flag; it now does so
  itself, which is where a plotting concern belongs.
- `Dynamics.to_lems()`, `SimulationExperiment.to_lems()` and
  `save_model_specification()` — use `NeuroMLAdapter(...)`, which emits validated
  XML rather than a `lems.Model` and covers constructs those never did.
- `Network.compute_delays()` → `calculate_delays()`;
  `Dynamics.add_coupling_term()` → `add_coupling_input()`; `Connectome` →
  `Network` (it was a subclass that added a warning and nothing else).
- `equation.piecewise2numpy()` / `piecewise2julia()` — use
  `codegen.code.render_expression(expr, format=...)`. `print_Piecewise` is the one
  handler every printer shares, emitting through each printer's `_where3`
  primitive (`<mod>.where` for numpy/JAX, `ifelse` for Julia). Both predated it,
  duplicated one backend each, and raised on any Piecewise with a symbolic
  condition — so neither could have had a working caller.

### Fixed
- One integration method, one canonical name, whatever the recipe spells it.
  `rk4` is a method the schema advertises and the tvboptim adapter accepted, but
  the ontology holding the method's symbolic update expression knew only
  `RungeKutta4thOrder` — so a recipe spelling it `rk4` ran on tvboptim and died in
  the tvb and jax templates on `'NoneType' object has no attribute 'equation'`.
  `tvbo.utils.integration_method` now resolves every spelling to the one name, and
  an unknown one raises instead of falling back. The spellings had been written out
  five times (the adapter and four templates), disagreeing about which they
  accepted, and two of the templates resolved an unknown method to `Euler` —
  silently integrating a fourth-order recipe by a first-order scheme. There is one
  table now, and `tvbo.adapters.tvboptim.solver_class` raises for a method tvboptim
  has no solver for.
- Heterogeneous tvboptim runs: `Node.id` is resolved as an identifier rather than
  a row index (weights, lengths and delays alike), `execution.random_seed` reaches
  each group's noise, routes are keyed by coupling name so two edges naming one
  coupling are one route, and the declared noise sigma is used instead of a
  hard-coded 0.01.
- Brian2: a `random` current-pulse edge gets its own per-edge subset mask, so two
  pulses onto one population no longer collapse onto a single mask and fraction.
- Julia: a recipe declaring `parameters: {sigma: ...}` produces an `SDEProblem`
  rather than a silently deterministic `ODEProblem`.
- Bifurcation: periodic-orbit waveforms are actually recorded
  (`save_sol_every_step`), aligned to the branch's step axis.
- An `Edge`'s scalar `weight`/`delay`/`distance` slots are read by every backend,
  not only by pyrates.
- `DirectOntologyAPI` no longer raises `'str' object has no attribute 'storid'` on
  entities whose `requires` mixes entity references with bare names. `tvbo:requires`
  (an ObjectProperty holding entities) and the NeuroML ingest's `requires` (an
  AnnotationProperty naming a quantity a ComponentType needs from its context,
  `surfaceArea`, `iCa`) share one owlready2 attribute, so the slot yields both kinds
  at once. `get_children` links only the references, which are the ones that have a
  node to link to.

### Breaking
- `coupling:` on a simulation experiment is gone as a *declaration*. Declare it
  under `network:`, where a coupling function has a connectivity to act over.
  `network.coupling` is keyed by name, so a recipe moves the block one level down
  and names it: `network: {coupling: {Linear: {...}}}`. An edge still names one by
  reference. `SimulationExperiment(coupling=...)` raises rather than being ignored.
  Reading is unaffected: `experiment.coupling` is a read-only property answering the
  network's first coupling, which is the one a backend expressing a single coupling
  renders — codegen asks `BaseAdapter` for it instead, so no template derives its own.
  `scripts/migrate_experiment_coupling.py` performs the move: it reports by default
  and rewrites under `--apply`, moving only the coupling's own lines so the rest of a
  hand-authored recipe is left byte for byte as written. It carries anchors onto the
  re-keyed line, expands a `network: *alias` into a `<<:` merge, and names the file
  to edit when the network arrives by `!include`. A site it cannot place
  unambiguously is reported and left alone.
- A key repeated in one YAML mapping is now an error rather than a silent
  last-one-wins override.
- A serialized entity's `requires` is a list of labels. It used to be a list of
  storids, which are opaque per-session integers that do not survive a reload.

## 0.2.1 – 2025-11-20
### Changed
- Dependency/test adjustments: skip heavy `tvb` & notebook tests in release workflow; optional deps not required for minimal install.

### Fixed
- Minor packaging workflow refinements preparing for PyPI publish (trusted publishing test exclusions).

### Notes
- Incremental release before broader 1.0.0 stabilization; semantic versioning pre-1.0: minor component changes still bump patch.
