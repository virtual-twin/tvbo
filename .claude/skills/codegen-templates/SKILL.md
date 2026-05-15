---
name: codegen-templates
description: "How TVBO's code generation works \u2014 template engines, backend dispatch\
  \ in tvbo/codegen/, and the contract for adding a new backend."
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

## Common pitfalls

- **Don't mix Mako and Jinja2** in the same template tree — pick the one the sibling uses.
- **Don't hard-code paths** inside templates; pass them through the templater.
- Generated code must round-trip through `black` cleanly (Python backends).
