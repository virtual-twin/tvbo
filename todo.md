# TODO

## Harmonize class names with `tvboptim`

- change tvbo's "ExplorationAxis" to more broad "DataAxis" to fit with tvboptim's framework. It also makes more senese and is in line with tvbo's generalization goal.

## Per Task Backend Support in yaml
- Currently we're selecting backends per runtime and we have defaults set in tvbo
- It would be more valid and correct, if we define the backend to run a task with in the yaml/metadata spec it self. There we have the Software component, so we can also directly pin the current version and environment after we ran it.
  1. Select backend/software to run with in yaml
  2. If no version/environment is specified, it can be then set from the current environment the experiment is executed in. So metadata gets enriched post-run to be shared correctly.
  3. Running SimulationExperiment should be possible in both ways, 1) exactly the same environment as specified in yaml, 2) own environment


## Revisit Heterogenous parameter specification

- What is the current status of Network.nodes and where are the parameters to be expected?
  - Node.dynamics.parameters or Node.parameters?


## Improve Observations

- If we change to the task-based approach (SED-ML interoperability), we might also define the Observation-Pipeline as DAG of Tasks. By that, it gets more aligned with the other Specs (e.g. Pipeline).
    - A tasks requires specification of input, function, output (like FunctionCall).

- Is DerivedObservation as concept really sound? Actually, it would be more minimal to use single Observation class, but to clarify what is the output dimensionality.
  - Is it still time-series or has it changed dimnensionality, i.e. was as a dimensionality reduction on time-dimension applied

So we need to find a generalizable way to describe these Observations as pipeline of tasks, what needs to be done to derive a certain observation from the raw timeseries.
- Is additional data needed (external observation)?
- What dimension is looked at?
- What data is selected?

Examples:
There are classical examples, however we need to find a solution to describe any potential Observation

- Mean timeseries
- Frequency Spectrum / Single Band-Power
- Correlation (FC, FCD)
- Projection (Matrix-Multiplication)
- Convolution


### Explorations: Observations should be also available in Explorations

So I want to setup:
- Observation (let's say just mean)
- Exploration axis (parameter sweep)
- Plot Observation over a


## Documentation

- [ ] Always describe both, 1) Python API for model specification, 2) Pure Yaml
    > Currently, it is not clear how to use python-API, e.g. for defining/adding Observations and pipelines. Also always using pure yaml is not intuitive enough for iteratively setting up experiments.


## Migrate to Pydantic

- [ ] Find out if there is any benefit of linkml gen-python instead of gen-pydantic. Do we really need both?
- [ ] Change import of metadata-classes, so we can import all from tvbo.classes or tvbo.schema.
    > Having a common import structure for classes with extra functionality (inherited from LinkML) and pure LinkML export would be nice, so it's not confusing from where to import certain classes.


## Linkml Yaml shorthands
- [ ] Investigate shorthands for specs.
    > - It is a little cumbersome to specify equations always with `rhs` (equation={'rhs':'x+y'}), since we most of the time only need to specify `rhs` attribute. However it is relevant to have a proper `Equation` class. We need different equation types (differential, etc.), for StateVariable (ODE, PDE, ...), DerivedVariable (just algebraic), etc.
    > - So shortcutting to `equation='x+y'`, which resolves into `equation={'rhs':'x+y'}` would be really useful. But it needs to be linkml-native. No monkey-patching or hacks.
    > - It would be great that we can set for each class, that has equation as property, so we can define axioms, which equation-type they expect. We need to find out, if this is possible.



## Exploration Space must be keyed

Currently,
exp.explorations["a_sweep"].space

is list.

[ExplorationAxis({'parameter': 'a', 'explored_values': [-2.0, -1.0, 0.0, 1.0, 2.0]})]


But we want to be able to change the space of a specific axis. therefore we need keys.


## Interopearbility

## Data Standards
Neurodata without Borders (NDWB)


## Enable backend spec in yaml/metadata for different SimulationExperiment Tasks
- We already have the software database, implement the SimulationExperiment.run() with no backend specified,
-  the default backend can be still tvboptim but in metadat defined already
- This will go towards no hardcoded asumption in python runtime for tvbo
- Allows SimulationEperiment with multiple backends per task
    - Running integration/exploration with tvboptim
    - Bifurcation analysis with julia


## Bifurcation result needs to be also xarray structure!
- selection of variables should be possible etc.


## Harmonize IRI resolution and DB metadata fetching across all classes
- Currently each runtime class handles `iri`-based sourcing differently:
    - `Coupling.__init__` auto-resolves via `_populate_from_ontology` after super().__init__
    - `DynamicalSystem.__init__` resolves via the registry before super().__init__ (loads full YAML and merges)
    - `Network` derives `name` from `iri` for `atlas` / `tractogram` but does no DB fetch
    - `SimulationExperiment.__init__` does its own backfill for nested dicts (parcellation/tractogram/atlas)
- Each class also has its own `from_db` / `from_file` / `from_ontology` factories with subtly different behavior.
- Needed: one canonical IRI resolution layer that any schema class can opt into:
    1. Single `_resolve_iri(iri, category) -> dict` helper (registry-first, ontology fallback).
    2. Consistent rule for "iri given, name missing/default" → load and merge (user kwargs win).
    3. Apply uniformly to Dynamics, Coupling, Tractogram, Parcellation/BrainAtlas, and any future class with an `iri` slot.
    4. Drop duplicate ad-hoc backfill code in `SimulationExperiment.__init__` and `Network.__init__` once the per-class path is uniform.


## Drop `use_ontology` / `_skip_ontology` flags once IRI handling is canonical
- Today there are ~37 occurrences across `tvbo/` of `use_ontology`, `_skip_ontology`, `_populate_from_ontology*` runtime flags that gate ontology backfill.
- Once IRI is the canonical way to declare a sourced component, the flag becomes redundant: **iri present → use ontology/DB data; iri absent → fully self-contained spec.**
- Override semantics should follow YAML/dict merge: ontology defaults are the base, user-provided fields override key by key. Example target:
    ```python
    Dynamics(iri='tvbo:ReducedWongWang', parameters={'a': {'value': 2}})
    # → loads all parameters/state_variables from ontology, then overrides only a.value
    ```
- Cleanup steps:
    1. Remove `use_ontology` / `_skip_ontology` parameters from `DynamicalSystem.__init__`, `Dynamics.from_*`, `Coupling.*` and any other class constructors.
    2. Remove the explicit `_populate_from_ontology_by_name()` / `_populate_from_ontology()` call sites — they become unconditional inside the single `_resolve_iri` step from the previous TODO.
    3. Ensure parameter/state-variable merging is non-destructive: user dict values overwrite at the leaf level (e.g. `parameters.a.value`), not the whole `parameters` slot.
    4. Update tests that pass `use_ontology=True/False` explicitly.
