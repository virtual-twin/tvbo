## Unreleased

### Added
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

### Deprecated
- `noise: {intensity: ...}` warns when read. It is interpreted as a standard
  deviation; use `parameters: {sigma: ...}`, or `parameters: {nsig: ...}` if the
  value was a dispersion (the two differ by sqrt(2D)/D).

### Fixed
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

### Breaking
- A key repeated in one YAML mapping is now an error rather than a silent
  last-one-wins override.

## 0.2.1 – 2025-11-20
### Changed
- Dependency/test adjustments: skip heavy `tvb` & notebook tests in release workflow; optional deps not required for minimal install.

### Fixed
- Minor packaging workflow refinements preparing for PyPI publish (trusted publishing test exclusions).

### Notes
- Incremental release before broader 1.0.0 stabilization; semantic versioning pre-1.0: minor component changes still bump patch.
