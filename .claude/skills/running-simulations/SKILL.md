---
name: running-simulations
description: "How to run a SimulationExperiment in TVBO \u2014 discovering curated\
  \ components, reusing a curated experiment, choosing a backend, and calling run/plot.\
  \ Read this before grepping the repo for models, networks, or how to run something."
---

# Running Simulations

Once you have a `Dynamics` (see the `writing-models` skill), wrap it in a
`SimulationExperiment` and run it. Three habits keep this cheap and correct —
**follow them before writing exploratory code or searching the repo**:

1. **Discover curated components** with the catalog API — don't grep the tree.
2. **Reuse a curated experiment** when one already matches — don't rebuild it.
3. **Build from `iri` pointers**, not from hand-copied parameter values.

## 1. Discover what's available — don't grep the repo

TVBO ships a large curated database (100+ models, connectomes, atlases, coupling
functions, and whole experiments). Query it directly instead of searching files:

```python
from tvbo import Dynamics
Dynamics.list_db()                       # every model name (100+)
Dynamics.list_db(model_type="mean_field")# filter: mean_field | neural_mass |
                                         # phase_oscillator | spiking | field | …
Dynamics.db_overview()                   # pandas DataFrame: name, type, description

from tvbo.data.registry import list_entries
list_entries("Network")                  # curated connectomes (dTOR, Schaefer, HCP, …)
list_entries("BrainAtlas")               # DesikanKilliany, Schaefer2018, Yeo17, hcpmmp1, …
list_entries("Coupling")                 # Linear, Sigmoidal, Difference, Kuramoto, …
list_entries("SimulationExperiment")     # ready-to-run experiments
list_entries("SimulationStudy")          # published studies (parameter sweeps, fits)
```

To inspect one entry without running it, use the CLI (no Python needed):

```bash
tvbo info dynamics:JansenRit             # parameters, state vars
tvbo info experiment:RWW_BOLD_FC_Optimization   # tasks, network, backends
tvbo info study:Cabral2011
```

`tvbo info` accepts a path, a CURIE (`dynamics:`, `network:`, `coupling:`,
`atlas:`, `experiment:`, `study:`), or a bare DB name.

## 2. Fastest path: reuse a curated experiment

If an experiment already exists, loading it is one call — no reconstruction:

```python
from tvbo import SimulationExperiment
exp = SimulationExperiment.from_db("RWW_BOLD_FC_Optimization")
result = exp.run("jax")
result.plot()
```

Or run it end-to-end from the shell:

```bash
tvbo run experiment:RWW_BOLD_FC_Optimization --backend jax
```

## 3. Build a whole-brain experiment from curated parts

When no curated experiment fits, assemble one from `iri` pointers. This recipe is
complete and runnable as-is — swap the names for any from step 1:

```python
from tvbo import SimulationExperiment

exp = SimulationExperiment(
    dynamics={"name": "ReducedWongWangExcInh", "iri": "tvbo:ReducedWongWangExcInh"},
    coupling={"name": "Linear", "iri": "tvbo:Linear"},
    network={"iri": "network:example_3node_network"},   # or any list_entries("Network") name
    integration={"method": "Heun", "duration": 500, "noise": None},
)
result = exp.run("jax")     # ExperimentResult
result.plot()               # built-in plotting
result.data                 # xarray DataArray, dims (time, state_variable, node)
```

For a real connectome, point `network` at a curated relmat, e.g.
`{"iri": "network:tpl-FSLMNI152_cohort-HCPYA_rec-dTOR_atlas-Schaefer2018_seg-17Networks_scale-100_desc-SC_relmat"}`.

### The single-node quick path

For a bare `Dynamics` with no network (single node), the minimal form still holds:

```python
from tvbo import Dynamics, SimulationExperiment
SimulationExperiment(dynamics=my_dynamics).run("jax").plot()
```

## Declarative sourcing: inline vs YAML vs IRI

TVBO is declarative about **where each component comes from**. A
`SimulationExperiment` section is always one of:

1. **Fully inline** — a Python instance or a dict containing every field.
2. **From YAML** — `SimulationExperiment.from_file(...)` / `from_string(...)`.
3. **Semantic pointer** — a dict with an `iri` CURIE. The prefix selects the
   source (`tvbo:` for the built-in ontology; in principle other prefixes
   like `neuroml:` can resolve from other sources); remaining fields are
   backfilled from that source.

A bare name string is **not** a semantic pointer — it has no prefix, so the
resolver cannot tell which source to query. Use the `iri` form. `coupling`
additionally accepts a bare string as a `tvbo:`-prefixed shorthand, but prefer
the explicit `iri` form for clarity.

## Choosing a backend: `run(format=…)` and `render_code(format=…)`

The backend is selected by the **`format`** argument, not a constructor
`backend=` kwarg:

```python
result = exp.run("jax")           # execute on JAX (recommended default for sims)
result = exp.run("tvb")           # execute on The Virtual Brain
code   = exp.render_code("jax")   # render code WITHOUT executing (export)
```

`run()`'s default `format` is `tvboptim` (the optimisation backend); pass an
explicit `format` such as `"jax"` for a plain forward simulation. `duration` can
be passed to `run()` to override the integration length.

| Backend (`format`) | Extras to install | Use when |
|---|---|---|
| `jax` | in core | Default forward sim; differentiable; fast on GPU/TPU |
| `tvboptim` | `pip install tvbo[tvboptim]` | JAX-based optimisation / parameter fitting |
| `tvb` | `pip install tvbo[tvb]` | TVB's curated neural-mass models & monitors |
| `pyrates` | `pip install tvbo[pyrates]` | Rate-coded models + numerical continuation |
| `julia` | `pip install tvbo[julia]` | DifferentialEquations.jl, ModelingToolkit, NetworkDynamics |
| `numpy` | in core | Reference / debugging |
| `neuroml` | `pip install tvbo[neuroml]` | Multi-compartment models via NEURON/LEMS |

Discover installed backends with `from tvbo.cli._backends import list_backends`.

## Zero-Python: the `tvbo` CLI

Everything above has a shell equivalent — often the cheapest way to run one thing:

```bash
tvbo run experiment:JansenRit-bifurcation --backend jax   # run a curated spec
tvbo run ./my_experiment.yaml                             # run a local YAML
tvbo info <SPEC>                                          # inspect without running
tvbo export jax <SPEC>                                    # render code, no execution
tvbo workflow snakemake <SPEC> -o ./kit                   # emit an HPC/pipeline kit
```

## Parameter sweeps, optimization & inference

Sweeps and gradient-based fits are **declared on the experiment** as
`explorations`, not hand-written as optimizer loops — TVBO generates the code and
runs it on the `tvboptim` backend (a JAX autodiff engine):

```python
exp = SimulationExperiment.from_db("RWW_BOLD_FC_Optimization")
result = exp.run("tvboptim")     # fit G to empirical FC, gradient-based
```

Reuse a curated optimization experiment as your starting point instead of wiring
one from scratch — confirmed examples in the database:

| Load with `from_db("…")` | What it does |
|---|---|
| `RWW_BOLD_FC_Optimization` | Fit global coupling to empirical BOLD FC |
| `JR_MEG_FrequencyGradient_Optimization` | Fit Jansen-Rit to a MEG frequency gradient |
| `EI_Tuning_FIC_EIB_Optimization` | Feedback-inhibition control / E-I balance tuning |
| `Hopf_Pareto_ParallelOpt` | Multi-objective (NSGA-II) Pareto optimization |
| `Stimulation_Bayesian_Inference` | Bayesian (MCMC) inference over a stimulation model |
| `TBPTT_JansenRit_FC_Optimization` | Truncated-BPTT fit of Jansen-Rit to FC |

`tvbo info experiment:<name>` shows an experiment's tasks and explorations before
you run it. See `tvbo/classes/` for the `Exploration` metadata slots.

## Studies

A `SimulationStudy` aggregates multiple experiments (e.g. parameter sweeps). Load
one with `SimulationStudy.from_db(name)` (see `list_entries("SimulationStudy")`),
or `tvbo run study:<name>`. See `tvbo/classes/study.py`.

## Pitfalls

- **Backend selection is `format=`, not `backend=`.** There is no `backend=`
  constructor argument on `SimulationExperiment`.
- **`run()` defaults to `tvboptim`.** For a plain forward simulation pass
  `run("jax")` explicitly unless you installed the `tvboptim` extra.
- **Intel Mac**: TVBO pins `numba<0.60` / `llvmlite<0.44` on `darwin/x86_64`, and
  JAX to `0.4.28` (last Intel-Mac release). Check `pyproject.toml`'s platform
  conditionals if installation complains.
- **Julia extra**: requires a working Julia runtime; `juliacall` downloads one on
  first import if absent.
- `--run-slow` is a *pytest* flag, not a TVBO option — it controls which tests
  run, not which simulations.
