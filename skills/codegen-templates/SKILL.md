---
name: codegen-templates
description: How TVBO's code generation works — template engines, backend dispatch in tvbo/codegen/, and the contract for adding a new backend.
metadata:
  audience: maintainer
  applies_to:
    - "tvbo/codegen/**"
    - "tvbo/templates/**"
    - "tvbo/adapters/**"
  tags: [codegen, templates, backends]
---

# Code Generation

TVBO compiles a YAML/Python `Dynamics` spec into runnable code for one of several backends. Two moving parts:

## 1. Templates — `tvbo/templates/`

Per-backend template trees:

- `generic_python/`, `autodiff/`, `pyrates/`, `julia/`, `modelingtoolkit/`, `networkdynamics/`, `neuroml/`, `numcont/`, `pde/`, `rateml/`, `modules/`, `base/`

Templates render with **Mako**. Black is used to post-format Python output (see `dependencies` in `pyproject.toml`). When adding a new template, copy the closest sibling directory and adjust — don't invent a new engine.

## 2. Dispatch — `tvbo/codegen/`

- `templater.py` — locates the right template tree for a backend.
- `code.py`, `functions.py`, `pyrates.py`, `lems.py`, `cuda.py` — backend-specific glue between the `Dynamics` AST and the templates.

The entry point a user typically hits is `SimulationExperiment(dynamics=...).render_code('jax')`.

## 3. Adapters — `tvbo/adapters/`

Adapters wrap *external* simulators (TVB, Julia, NeuroML, PyRates, BifurcationKit, ModelingToolkit, NetworkDynamics, openMINDS, tvboptim). The contract lives in `tvbo/adapters/base.py`.

## Adding a backend

1. Add the adapter under `tvbo/adapters/<backend>.py` extending the base class.
2. Add the template tree under `tvbo/templates/<backend>/`.
3. Add a `backend_<backend>` pytest marker in `[tool.pytest.ini_options].markers` (see the `tests-and-backends` skill).
4. Wire dispatch in `tvbo/codegen/templater.py`.
5. Ensure the optional dependency lives in `pyproject.toml` under `[project.optional-dependencies]`.

## Slim templates — logic lives in the adapter

**Templates fill in values; they do NOT process.** All logic — resolution, parsing, AND generating code fragments — belongs in the Python adapter layer (`tvbo/adapters/<backend>.py` `prepare_context()`, or a helper it owns such as `tvbo/templates/tvboptim/utils.py`). The adapter returns ready-to-emit strings/context; the template just interpolates `${...}`.

- If a `<% %>` block branches over metadata to emit different code bodies (e.g. per-type `% if/elif` building a function body), move it to a Python `render_*()` helper and interpolate the result. A verbatim function body with heavy `% if/for` branching inside a template is the smell.
- **Why:** the adapter can dedup, reduce redundancy, and harmonize; it also surfaces which generator functions can be **reused across adapters** (jax/julia/matlab). Logic locked in one backend's mako can't be shared. Python helpers are also testable in isolation.
- Good pattern: `resolve_solver_kwargs` / `resolve_optimizer_mode` in `utils.py` — pure functions returning strings, called once from the template.
- This also covers resolution/parsing: stringly-typed pointers (`source: [network.observations.X]`) get resolved ONCE at load into typed data, not re-parsed in template + utils + runtime.

## Common pitfalls

- **Don't mix Mako and Jinja2** in the same template tree — pick the one the sibling uses.
- **Don't hard-code paths** inside templates; pass them through the templater.
- **Don't cram processing into `<% %>` blocks** — see "Slim templates" above.
- Generated code must round-trip through `black` cleanly (Python backends).
