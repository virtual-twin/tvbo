---
name: running-simulations
description: "How to run a SimulationExperiment in TVBO \u2014 choosing a backend,\
  \ calling run/plot, and what each optional extra (jax, tvb, pyrates, julia) provides."
---

# Running Simulations

Once you have a `Dynamics` (see the `writing-models` skill), wrap it in a `SimulationExperiment` and run it.

## The quick path

```python
from tvbo import Dynamics, SimulationExperiment

experiment = SimulationExperiment(dynamics=my_dynamics)
experiment.run(duration=1000).plot()
```

`run()` returns a result container; `plot()` is built-in. For details see `tvbo/run/`, `tvbo/plot/`, `tvbo/analysis/`.

## Declarative sourcing: inline vs YAML vs IRI

TVBO is declarative about **where each component comes from**. A
`SimulationExperiment` section is always one of:

1. **Fully inline** — a Python instance or a dict containing every field.
2. **From YAML** — `SimulationExperiment.from_file(...)` / `from_string(...)`.
3. **Semantic pointer** — a dict with an `iri` CURIE. The prefix selects the
   source (`tvbo:` for the built-in ontology; in principle other prefixes
   like `neuroml:` can resolve from other sources); remaining fields are
   backfilled from that source.

A bare name string is not a semantic pointer — it has no prefix, so the
resolver cannot tell which source to query. Use the `iri` form:

```python
exp = SimulationExperiment(
    dynamics={"name": "ReducedWongWang", "iri": "tvbo:ReducedWongWangExcInh"},
    coupling={"name": "Linear", "iri": "tvbo:Linear"},
    network={
        "parcellation": {"atlas": {"name": "DesikanKilliany",
                                    "iri": "tvbo:DesikanKilliany"}},
        "tractogram":   {"name": "dTOR", "iri": "tvbo:dTOR"},
    },
    integration={"method": "Heun", "duration": 10_000, "noise": None},
)
```

`coupling` additionally accepts a bare string as a `tvbo:`-prefixed shorthand,
but prefer the explicit `iri` form for clarity and to avoid coupling docs to
the default namespace.

## Picking a backend

The default backend is JAX. Switch via `render_code('<backend>')` for export, or pass `backend=...` to `SimulationExperiment`.

| Backend | Extras to install | Use when |
|---------|------------------|----------|
| `jax` | `pip install tvbo[jax]` (already in core) | Default; differentiable; fast on GPU/TPU |
| `tvb` | `pip install tvbo[tvb]` | You need TVB's curated set of neural-mass models |
| `pyrates` | `pip install tvbo[pyrates]` | Rate-coded brain network models + numerical continuation |
| `julia` | `pip install tvbo[julia]` | DifferentialEquations.jl, ModelingToolkit, NetworkDynamics |
| `neuroml` | `pip install tvbo[neuroml]` | Multi-compartment models via NEURON/LEMS |
| `tvboptim` | `pip install tvbo[tvboptim]` | JAX-based optimisation / parameter fitting |

## Render-only (no execution)

```python
code = experiment.render_code('jax')
print(code)
```

Useful for inspecting the generated code or running it outside TVBO.

## Studies

A `SimulationStudy` aggregates multiple experiments (e.g. parameter sweeps). See `tvbo/classes/study.py`.

## Pitfalls

- **Intel Mac**: TVBO pins `numba<0.60` and `llvmlite<0.44` on `darwin/x86_64`. JAX is pinned to `0.4.28` (last Intel-Mac release). If installation complains, check `pyproject.toml`'s platform conditionals.
- **TVB extra**: pulls `numba` transitively; the same Intel-Mac caps apply.
- **Julia extra**: requires a working Julia runtime. `juliacall` will download one on first import if absent.
- `--run-slow` is a *pytest* flag, not a TVBO option. It controls which tests run, not which simulations.
