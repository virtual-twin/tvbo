---
name: running-simulations
description: How to run a SimulationExperiment in TVBO — discovering curated components, reusing a curated experiment, choosing a backend, and calling run/plot. Read this before grepping the repo for models, networks, or how to run something.
metadata:
  audience: user
  applies_to:
    - "**/*.py"
    - "**/*.ipynb"
  tags: [simulation, backends, jax, tvb, discovery]
  requires_extras: []
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

Dynamics.list_db()  # every model name (100+)
Dynamics.list_db(model_type="mean_field")  # filter: mean_field | neural_mass |
# phase_oscillator | spiking | field | …
Dynamics.db_overview()  # pandas DataFrame: name, type, description

from tvbo.data.registry import list_entries

list_entries("Network")  # curated connectomes (dTOR, Schaefer, HCP, …)
list_entries("BrainAtlas")  # DesikanKilliany, Schaefer2018, Yeo17, hcpmmp1, …
list_entries("Coupling")  # Linear, Sigmoidal, Difference, Kuramoto, …
list_entries("SimulationExperiment")  # ready-to-run experiments
list_entries("SimulationStudy")  # published studies (parameter sweeps, fits)
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
    # Coupling is declared on the network, keyed by name — it acts over a connectivity.
    network={
        "iri": "network:example_3node_network",  # or any list_entries("Network") name
        "coupling": {"Linear": {"name": "Linear", "iri": "tvbo:Linear"}},
    },
    integration={"method": "Heun", "duration": 500, "noise": None},
)
result = exp.run("jax")  # ExperimentResult
result.plot()  # built-in plotting
result.data  # xarray DataArray, dims (time, state_variable, node)
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
result = exp.run("jax")  # execute on JAX (recommended default for sims)
result = exp.run("tvb")  # execute on The Virtual Brain
code = exp.render_code("jax")  # render code WITHOUT executing (export)
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

## Running at scale: the `tvbo workflow` family

For a cohort or a heavy fit that won't fit one process, the **same** experiment/study
emits a self-contained cluster kit — no driver, no hand-written sbatch:

```bash
tvbo workflow plan      <SPEC>                  # show the resolved DAG (no artefact emitted)
tvbo workflow snakemake <SPEC> -o ./kit --pack  # emit the whole study as one Snakemake DAG, one .tar.gz
tvbo workflow submit    ./kit.tar.gz            # run it (engine auto-detected); --dry-run validates first
```

`snakemake` / `slurm` / `nextflow` choose the engine; one rule per experiment, dataset
experiments fan out per subject, and a `from_experiment` dependency becomes a DAG edge.
The **run environment is declared in the recipe's `workflow:` block** — `container:
docker://…` runs every rule inside that image via Apptainer (no venv / module load),
per-subject inputs travel via `Dataset.bundle: true`, and per-rule resources
(`cpus_per_task` / `mem` / `time` / `partition`) via `workflow.slurm`. So the recipe that
runs locally scales out unchanged.

`--dry-run` only resolves the DAG — it does **not** execute a rule, so it cannot catch a
runtime bug; smoke-test one experiment in the container before launching a whole cohort.
For the full scale-out discipline (compute-node orchestration, the read-only-container
bug class + binds, streaming observables for long fits), see the **replicating-studies**
skill, Phase 8.

## Parameter sweeps, optimization & inference

Sweeps and gradient-based fits are **declared on the experiment** as
`explorations`, not hand-written as optimizer loops — TVBO generates the code and
runs it on the `tvboptim` backend (a JAX autodiff engine):

```python
exp = SimulationExperiment.from_db("RWW_BOLD_FC_Optimization")
result = exp.run("tvboptim")  # fit G to empirical FC, gradient-based
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

## Before you trust a number: converge the *statistic*, not the trajectory

A run that looks right can still carry enough integration error to change what you
report. The trap is that the two things have very different sensitivity: a
**derived statistic** — an argmax, a peak latency, a threshold crossing, an event
count, any extremum — can move by a large factor while the trajectory it is read
from barely moves at all.

A measured case: the same linear equation, same basis, same stimulus, same
parameters, integrated with an adaptive solver at its **default** tolerances
(`rel_tol` 1e-3 / `abs_tol` 1e-6) versus tight ones (1e-8 / 1e-10). The
trajectories agree at r ≈ 0.998. The statistic actually being reported — a
per-region peak latency — flagged **24–26** anomalous regions at the default
tolerance against **3** at the tight one, and the headline correlation moved from
−0.19 to −0.41. Nothing warned; both runs completed cleanly.

So converge on the number you will publish:

```python
base = exp.run("jax")

exp.integration.step_size /= 4  # fixed-step (Heun, RK4): halve/quarter dt
exp.integration.rel_tol = 1e-12  # adaptive (Tsit5, Vern9, Rodas5): tighten
exp.integration.abs_tol = 1e-12
refined = exp.run("jax")

# compare YOUR statistic, not ||trajectory||
assert abs(my_metric(refined) - my_metric(base)) < a_tolerance_you_can_defend
```

`exp.integration` is an `Integrator`; `method`, `step_size`, `rel_tol`, `abs_tol`
and `duration` are plain assignable fields on it. Which knob bites depends on the
method — tolerances do nothing for a fixed-step `Heun`, `step_size` does nothing
for an adaptive one.

Three habits that make this cheap:

- **Refine until the reported number stops moving**, using the knob your method
  actually reads. TVBO defaults `rel_tol` / `abs_tol` to **1e-10** — far tighter
  than most external tools (MATLAB `ode45` ships 1e-3 / 1e-6), so a disagreement
  with another package is often a comparison of tolerances rather than of models.
  Check theirs before concluding anything about yours.
- **Watch adaptive solvers around brief inputs.** An adaptive solver sizes its
  step from local error, and through a quiescent stretch before a stimulus that
  error is ≈ 0 — so it can take one enormous step straight over a short pulse and
  never notice. Cap the step, or use a fixed step no larger than the input's
  width. In the case above, most modes had a natural period shorter than the
  solver's own default maximum step.
- **Prefer an oracle to another integrator.** Where the system admits a closed
  form — a linear model under a step, pulse or sinusoidal drive usually does —
  compare against *that*; it settles in one line what a solver-versus-solver
  comparison cannot, because two integrators can be wrong in the same direction.
  Otherwise cross-check two backends (`run("jax")` vs `run("numpy")` /
  `run("julia")`), which is part of what the `format=` split is for.

The same discipline covers any other resolution knob. For a modal or field model
the **mode count is a model parameter, not a detail**: sweep it and report the
sensitivity rather than trusting one truncation. For regime-dependent step-size
traps (a step sized for one parameter value that a sweep then leaves behind), see
the symptom index in the **replicating-studies** skill.

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
