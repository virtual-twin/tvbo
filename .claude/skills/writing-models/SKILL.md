---
name: writing-models
description: "How to specify a Dynamics in TVBO \u2014 the YAML and Python forms,\
  \ parameter / state-variable / equation conventions, and common pitfalls."
---

# Writing Models in TVBO

A **Dynamics** is the smallest building block: a set of named parameters and state variables governed by ODE equations. It can be written as YAML or constructed directly in Python.

## YAML form

```yaml
name: LorenzAttractor
parameters:
    sigma:
        value: 10
        label: Prandtl number
    rho:
        value: 28
        label: Rayleigh number
    beta:
        value: 2.6666666666666665
state_variables:
    X:
        equation:
            lhs: \dot{X}
            rhs: sigma * (Y - X)
    Y:
        equation:
            lhs: \dot{Y}
            rhs: X * (rho - Z) - Y
    Z:
        equation:
            lhs: \dot{Z}
            rhs: X * Y - beta * Z
```

## Python form

```python
from tvbo import Dynamics

lorenz = Dynamics(
    parameters={
        "sigma": {"value": 10.0},
        "rho": {"value": 28.0},
        "beta": {"value": 8 / 3},
    },
    state_variables={
        "X": {"equation": {"rhs": "sigma * (Y - X)"}},
        "Y": {"equation": {"rhs": "X * (rho - Z) - Y"}},
        "Z": {"equation": {"rhs": "X * Y - beta * Z"}},
    },
)
```

## Loading a curated model from a source

TVBO components are declarative: a `Dynamics` is either specified inline (as
above), loaded from YAML, or **pointed at semantically via an `iri`**. The
IRI's prefix names the source — `tvbo:` for the built-in ontology, but the
same mechanism is intended to dispatch to other prefixes (e.g. `neuroml:`)
that resolve from other ontologies / data sources.

```python
# Direct construction (Python API)
dyn = Dynamics.from_db("ReducedWongWangExcInh")

# As a semantic pointer inside a SimulationExperiment dict
dynamics = {"name": "ReducedWongWang", "iri": "tvbo:ReducedWongWangExcInh"}
```

A bare name string (`dynamics="ReducedWongWangExcInh"`) is **not** a semantic
pointer — there's no prefix, so the resolver cannot tell which source to
query. Always include the `iri`.

## Conventions and pitfalls

- **`lhs` is LaTeX**, not Python. `\dot{X}` for a time-derivative. Omit `lhs` to default to `\dot{<state>}`.
- **`rhs` is a SymPy-parseable expression**. Names must match parameters and state variables. Greek letters as full words: `sigma`, not `σ`.
- **`label`** on a parameter is human-readable metadata (renders in the platform browser). Keep `name` machine-friendly.
- Parameters carry **values**; state variables carry **equations**. Don't put a `value` on a state variable.
- **Don't hand-edit `tvbo/datamodel/**`** — that's generated from `schema/*.yaml`. Use the `Dynamics` class.
- **Trust the slots. Never assign a raw `{}` / `[]` to a LinkML slot — a `JsonObj` in your data is the symptom of mis-configured YAML / a Python object leaking into the schema.** The active datamodel (`tvbo/datamodel/schema.py`) is the jsonasobj2 / `YAMLRoot` dataclass form. Inlined dicts/lists are coerced into typed classes **only in `__post_init__` (i.e. at construction)**. Mutating a slot afterwards does **not** re-run that coercion:
    - **Multivalued slots** (dict- or list-backed: `parameters`, `dynamics`, `coupling`, `nodes`, `edges`, `transforms`, …) → **mutate the existing container in place**: `.update(...)` / `container["X"] = ...` for dict slots, `.append(...)` for list slots. To get a *typed* entry, insert a constructed class: `net.dynamics["X"] = Dynamics(...)`.
        - ❌ `net.dynamics = {...}` — a raw dict goes through jsonasobj2's `JsonObj.__setattr__`, which wraps it in a bare `JsonObj` (it later breaks `.items()` / `.values()` on round-trip).
        - ❌ `net.nodes = [{...}]` — a raw list isn't wrapped in `JsonObj`, but its entries stay **uncoerced plain dicts**, never `Node` instances.
        - ⚠️ `.update({...})` / `["X"] = {...}` avoids the `JsonObj`, but a raw dict still isn't normalized — prefer inserting a constructed class instance.
    - **Single-valued slots** → plain `=` is correct. Assign a scalar directly (`net.number_of_nodes = 76`); for a nested-object slot assign a **constructed LinkML class instance** (`net.coordinate_space = CommonCoordinateSpace(...)`), never a raw `{}` (which becomes a bare `JsonObj`).
    - **Best of all:** pass everything at construction (`Network(dynamics={...}, nodes=[...])`) so `__post_init__` normalizes it into typed classes for you.
- **Never change a datatype; only assign valid LinkML class instances.** If a slot expects a class, construct that class — don't pass a bare dict or an ad-hoc object.
    > This holds as long as the active datamodel is the dataclass/`YAMLRoot` one. If TVBO switches to the pydantic datamodel (`tvbo/datamodel/pydantic.py`), dict-assignment may become valid (pydantic validates/coerces) — unverified.

## Beyond ODEs

- **Stochastic** terms via `tvbo.classes.noise`.
- **Discrete events** via `tvbo.classes.event`.
- **Perturbations / stimulation** via `tvbo.classes.perturbation`.
- **Network coupling** via `tvbo.classes.coupling` + `tvbo.classes.network`.
- **Continuation** (bifurcation analysis) via `tvbo.classes.continuation`.

See the `running-simulations` skill for what to do once you have a `Dynamics`.
