---
name: writing-models
description: How to specify a Dynamics in TVBO — the YAML and Python forms, parameter / state-variable / equation conventions, and common pitfalls.
metadata:
  audience: user
  applies_to:
    - "**/*.yaml"
    - "**/*.yml"
    - "**/*.py"
  tags: [models, dynamics, yaml]
  requires_extras: []
---

# Writing Models in TVBO

A **Dynamics** is the smallest building block: a set of named parameters and state variables governed by ODE equations. It can be written as YAML or constructed directly in Python.

## First: is it already curated? (don't rewrite from a paper)

TVBO ships 100+ curated models. Check before transcribing equations by hand — a match is one `from_db` call and is already unit-checked:

```python
from tvbo import Dynamics

Dynamics.list_db()  # every curated model name
dyn = Dynamics.from_db("JansenRit")  # load one
```

Only write a new `Dynamics` when the catalog has no match. See the `running-simulations` skill for the rest of the discovery API (filtering by `model_type`, `db_overview`, the `tvbo info` CLI) and for finding networks, atlases, coupling, and whole curated experiments the same way.

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

TVBO components are declarative: a `Dynamics` is either specified inline (as above), loaded from YAML, or **pointed at semantically via an `iri`**.

```python
# Direct construction (Python API)
dyn = Dynamics.from_db("ReducedWongWangExcInh")

# As a semantic pointer inside a SimulationExperiment dict
dynamics = {"name": "ReducedWongWang", "iri": "tvbo:models/ReducedWongWangExcInh"}
```

Always include the `iri` — a bare name string is **not** a semantic pointer. See the `running-simulations` skill for how a prefix resolves to a source and for the full sourcing rules (inline vs YAML vs `iri`).

## Conventions and pitfalls

- **`lhs` is LaTeX**, not Python. `\dot{X}` for a time-derivative. Omit `lhs` to default to `\dot{<state>}`.
- **`rhs` is a SymPy-parseable expression**. Names must match parameters and state variables. Greek letters as full words: `sigma`, not `σ`.
- **`label`** on a parameter is human-readable metadata (renders in the platform browser). Keep `name` machine-friendly.
- Parameters carry **values**; state variables carry **equations**. Don't put a `value` on a state variable.
- **Define models through the `Dynamics` class or YAML** — never by editing the generated datamodel under `tvbo/datamodel/`.
- **Trust the slots.** The datamodel classes are Pydantic models that validate on construction and on assignment, so a slot holds typed members however it was filled: `Network(dynamics={...}, nodes=[...])` builds typed members, `net.dynamics = {...}` validates the mapping into them, and an invalid value raises a `ValidationError` instead of being stored. Inserting into a keyed collection (`net.dynamics["X"] = Dynamics(...)`) takes a constructed member, since the mapping itself is a plain `dict`.
- **Never change a datatype; only assign what the slot declares.** If a slot expects a class, construct that class or pass the mapping it validates from — never an ad-hoc object.

## Beyond ODEs

- **Stochastic** terms via `tvbo.classes.noise`.
- **Discrete events** via `tvbo.classes.event`.
- **Perturbations / stimulation** via `tvbo.classes.perturbation`.
- **Network coupling** via `tvbo.classes.coupling` + `tvbo.classes.network`.
- **Continuation** (bifurcation analysis) via `tvbo.classes.continuation`.

See the `running-simulations` skill for what to do once you have a `Dynamics`.
