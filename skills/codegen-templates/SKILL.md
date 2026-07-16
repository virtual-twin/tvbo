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
3. Register a `backend_<backend>` pytest marker (see the `tests-and-backends` skill).
4. Wire dispatch in `tvbo/codegen/templater.py`.
5. Ensure the optional dependency lives in `pyproject.toml` under `[project.optional-dependencies]`.

## Slim templates — resolution in the adapter, code *structure* in Mako

Two separate concerns, two separate homes — don't merge them:

**1. Resolution / parsing → the Python adapter layer** (`tvbo/adapters/<backend>.py`
`prepare_context()`, or a helper it owns such as `tvbo/templates/tvboptim/utils.py`).
Turn stringly-typed metadata into clean, typed context ONCE: resolve dotted refs to
state paths, decode transforms, compute bounds, look up observation names. `source:
[network.observations.X]` is resolved at load into typed data, not re-parsed in
template + utils + runtime. Pure functions returning small strings/dicts (e.g.
`resolve_solver_kwargs`, `resolve_optimizer_mode`) are ideal — testable in isolation.

**2. Code structure / layout → a Mako `<%def>` partial, NOT Python string-building.**
The *shape* of generated code (a whole function body, a class, a branching block) is laid
out with Mako `% for` / `${...}` over the clean context — never assembled by a Python
helper doing `emit()`/`"".join()`/string concatenation. Building a 30–60-line body in
Python is the anti-pattern (fragile indentation, unreadable, un-diffable). Reserve Python
helpers for resolution and *small* one-line fragments; move anything structural to Mako.

**Modular partials** — for non-trivial or conditional codegen, factor it out of the
monolithic experiment template into its own partial and insert it conditionally (keeps the
big template readable and the feature self-contained):

```mako
## top of the experiment template
<%namespace name="search" file="tvbo-tvboptim-search.py.mako"/>
## imports at module scope, gated by a has_<feature> flag
% if has_nsga2:
import numpy as _np
from pymoo.optimize import minimize as _pymoo_minimize
% endif
## conditional insertion where the code belongs
% if expl['strategy'] == 'nsga2':
${search.nsga2_body(expl)}
% endif
```

The partial holds parameterized `<%def name="nsga2_body(expl)">…</%def>` blocks that lay
out the code from `expl` (already resolved: `expl['nsga2_axes']` with `path`/`transform`,
etc.). Reference implementation: `tvbo/templates/tvboptim/tvbo-tvboptim-search.py.mako`
(NSGA-II + Pareto-seed refinement). Heavy imports gated by `has_<feature>` at the module
top — never `import` inside a generated function body.

**Why:** the adapter can dedup/harmonize and its resolution reuses across backends
(jax/julia/matlab); the Mako partial keeps codegen readable and maintainable. A verbatim
function body with heavy `% if/for` branching buried in the 3000-line experiment template,
*or* a Python routine string-building that same body, are both smells — split them.

## Common pitfalls

- **Don't mix Mako and Jinja2** in the same template tree — pick the one the sibling uses.
- **Don't hard-code paths** inside templates; pass them through the templater.
- **Don't cram processing into `<% %>` blocks** — see "Slim templates" above.
- Generated code must round-trip through `black` cleanly (Python backends).
