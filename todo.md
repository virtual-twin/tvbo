# TODO

## LinkML 1.11.0 release follow-up

**Status:** workaround in place. Triggered by missing PyPI release of LinkML
1.11, which contains the SimpleDict-shorthand fixes we depend on.

### Why the workaround exists

`tvbo/database/models/*.yaml` uses the `Function` class with the SimpleDict
shorthand for `arguments`:

```yaml
H:
  arguments:
    - name: x
  expression: 1 / (1 + exp(-x))
```

LinkML 1.10.0 (current PyPI release) has two bugs that break this:

1. `linkml_runtime.utils.yamlutils._normalize_inlined` raises on
   `[{name: v}]` items for inlined-as-list slots inheriting from
   `Function`.
2. The jsonschema generator emits a schema that rejects the SimpleDict
   shorthand at validation time.

Both are fixed by these commits, all included in tag `v1.11.0-rc1`
(`0fa6f931e9edfc10a6410885e88eed496f6b9249`):

- `a5938c6d3` (runtime: scalar inlined-as-list keys)
- `3f2445d3b` (runtime: kwargs for inherited classes)
- `c3e37be08`, `c43a220ec` (jsonschema generator: SimpleDict support)

### Current pin

We install `linkml` and `linkml-runtime` from the LinkML monorepo at tag
`v1.11.0-rc1`, subdirectories `packages/linkml` and
`packages/linkml_runtime`.

The pin lives in three places:

1. `pyproject.toml` — `[tool.uv.sources]` block (used by `uv sync` /
   `uv pip install` of the project tree).
2. CI workflows — explicit `git+https://...@v1.11.0-rc1` URLs in
   every install step (the `uv pip` interface does not honor
   `[tool.uv.sources]`):
   - `.github/workflows/ci.yml` — `schema-validate`, `schema-artifacts`,
     `reasoner`, `compat` (cache-miss), `test-native` (cache-miss).
   - `.github/workflows/publish-pypi.yml` — `tests` job.
   - `.github/workflows/docker.yml` — `release-ready` wheel verification.
3. Docker images — `docker/Dockerfile` and `docker/Dockerfile.dev`
   install the same git URLs before reading `pyproject.toml`.

### Action items when LinkML 1.11.0 ships on PyPI

1. Bump version floors in `pyproject.toml`:
   - `"linkml-runtime>=1.8.5"` -> `"linkml-runtime>=1.11.0"`
   - `"linkml>=1.8.5"` -> `"linkml>=1.11.0"`
2. Remove the `[tool.uv.sources]` block from `pyproject.toml`.
3. Remove every `git+https://github.com/linkml/linkml.git@v1.11.0-rc1`
   install line from the workflow files listed above. Replace with a
   plain `uv pip install linkml linkml-runtime` (where needed) or rely
   on the version floor through `uv pip install -e .`.
4. Remove the `linkml @ git+...` / `linkml-runtime @ git+...` install
   step from `docker/Dockerfile` and `docker/Dockerfile.dev`.
5. Drop the `# NOTE: ... v1.11.0-rc1` comment blocks added next to each
   workaround.
6. Re-run `make gen-linkml` to confirm no regenerated drift, then
   `pytest tests/test_database_validation.py -q` (must remain 299/299).
7. Delete this section of `todo.md`.

### Verification command after removal

```bash
source .venv/bin/activate
uv pip install -e . --reinstall
pytest tests/test_database_validation.py -q
```

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


## Enable backend spec in yaml/metadata for different SimulationExperiment Tasks
- We already have the software database, implement the SimulationExperiment.run() with no backend specified,
-  the default backend can be still tvboptim but in metadat defined already
- This will go towards no hardcoded asumption in python runtime for tvbo
- Allows SimulationEperiment with multiple backends per task
    - Running integration/exploration with tvboptim
    - Bifurcation analysis with julia
