# Copilot Coding Agent Instructions for TVBO

## Code Organization Principles

have a look at skills folder for context

### Where to Put Code

1. **General Python Classes/Functions:** Add to `tvbo/` source code (e.g., `tvbo/data/types.py`, `tvbo/utils.py`)
   - Result classes (`SimulationResult`, `AlgorithmResult`, etc.) → `tvbo/data/types.py`
   - Utility functions used across modules → `tvbo/utils.py`
   - Data loading and processing → `tvbo/data/`

2. **Template Utilities (metadata processing):**
   - Language-agnostic utilities → `tvbo/templates/` (shared across backends)
   - Backend-specific utilities → `tvbo/templates/<subfolder>/utils.py` (e.g., `tvbo/templates/tvboptim/utils.py`)

3. **Templates (code generation):**
   - Templates generate code by combining YAML metadata with Mako syntax
   - Templates should import reusable classes from tvbo source, not define them inline
   - Use `<%namespace>` for shared template macros

### Template vs Source Code Decision Tree

```
Is this code specific to generated output syntax?
├── YES → Put in template (.py.mako)
└── NO → Is this utility for processing YAML metadata?
    ├── YES → Is it backend-specific?
    │   ├── YES → tvbo/templates/<backend>/utils.py
    │   └── NO → tvbo/templates/utils.py
    └── NO → tvbo/ source code (appropriate module)
```

### Examples

| Code Type | Location |
|-----------|----------|
| `SimulationResult` class | `tvbo/data/types.py` |
| `safe_name()` for template variable escaping | `tvbo/templates/tvboptim/utils.py` |
| JAX code printer | `tvbo/export/code.py` |
| Dynamics model template | `tvbo/templates/tvboptim/tvbo-tvboptim-dfun.py.mako` |
| Bunch utility class | `tvbo/utils.py` |

## Code Style Principles

1. **Minimal & Clean:** Write the shortest code that solves the problem. No boilerplate.
2. **No Redundancy:** Always check if logic already exists before defining new functions. Reuse existing code.
3. **MVP First:** No fallbacks, no try-except blocks. Code should work as expected; if it breaks, we debug.
4. **Readable:** Clear variable names, simple control flow, no unnecessary abstractions.

## Schema-Driven Development (CRITICAL)

**100% Trust the Schema. No Manual Unpacking.**

Both Odoo models and Pydantic models are generated from the same LinkML schema. This means:

1. **No Field-by-Field Unpacking:** Never write code like:
   ```python
   # WRONG - manual unpacking
   if record.parameters:
       data['parameters'] = record.parameters.read()
   if record.state_variables:
       data['state_variables'] = record.state_variables.read()
   # ... more fields
   ```

2. **Use Generic Schema-Driven Resolution:** Instead, iterate over the schema:
   ```python
   # CORRECT - schema-driven, adapts automatically when schema changes
   for field_name, field_obj in record._fields.items():
       if field_obj.type == 'many2one':
           data[field_name] = resolve_record(getattr(record, field_name))
       elif field_obj.type == 'many2many':
           data[field_name] = [resolve_record(r) for r in getattr(record, field_name)]
   ```

3. **Breaks Are Good:** If schema changes and code breaks, that's a FEATURE not a bug. Silent failures hide schema inconsistencies.

4. **Zero Redundancy:** If both Odoo and Pydantic have the same field structure, conversion should be automatic - not manually coded for each field.

5. **Single Source of Truth:** The LinkML schema (`schema/tvbo_datamodel.yaml`) defines everything. Code adapts to schema, not the other way around.

## Declarative Data Access Principles

**Data access should mirror YAML schema structure.** Users access results using the same path as the YAML definition:

1. **Schema-Aligned Access:** Result structure mirrors YAML sections. E.g., `results.integration`, `results.algorithms.fic`, `results.optimization.loss_fc`.
2. **No Implementation Details:** Never expose internal mechanisms via underscore prefixes (e.g., `._raw`). Users access `.data`, `.observations`, not internal representations.
3. **Wrap at Boundaries:** Internal functions use raw data types; only final user-facing returns wrap in result classes (`SimulationResult`, `AlgorithmResult`, etc.).
4. **Consistent Nesting:** `results.integration.observations.bold` - observations attached to the simulation that produced them.
5. **Convenience Aliases:** For common access patterns, provide both nested and flat access: `results.algorithms.fic` and `results.fic`.

Example - YAML to Python access:
```yaml
# YAML
integration:
  duration: 600000
algorithms:
  fic:
    n_iterations: 100
```

```python
# Python - mirrors YAML structure
results.integration.data              # Simulation data
results.integration.observations.bold # BOLD from main simulation
results.integration.transient         # Transient (warm-up) simulation
results.algorithms.fic.state          # FIC tuned state
results.algorithms.fic.history        # Per-iteration tracking
```

## Template & Code Generation Principles

**100% Generalizability is mandatory.** TVBO templates must work for ANY simulation, algorithm, exploration, or optimization:

1. **No Hardcoded Names:** Never mention specific parameter names (e.g., `J_i`, `S_e`), variable names, or model-specific logic in templates. All names must come from YAML metadata via template variables.
2. **No Special Cases:** No `if parameter_name == 'J_i'` or similar conditionals. Logic must be generic and driven by schema attributes (e.g., `is_coupling_param`, `has_pipeline`).
3. **Schema-Driven:** All behavior differences must be expressible through YAML schema attributes, not hardcoded in templates.
4. **Consistent Data Structures:** Ensure arrays/scalars are handled uniformly. If a parameter can be per-node, initialize it as an array from the start.
5. **Template Variables Only:** Use `${parameter_name}`, `${obs_name}`, etc. Never write literal parameter/variable names in generated code patterns.

Example - WRONG:
```mako
# Don't do this - hardcoded parameter name
result_history.J_i.append(state.dynamics.J_i)
```

Example - CORRECT:
```mako
# Do this - uses template variable
result_history.${target_name}.append(state.dynamics.${target_name})
```

## Mako Template Best Practices

**Minimal processing inside templates.** Templates should be clean and redundancy-free:

1. **Metadata Objects First:** Information is already well-structured in LinkML metadata objects (`model.parameters`, `coupling.parameters`, etc.). Access attributes directly rather than transforming.

2. **Shared Utilities in `utils.py`:** Common extraction patterns belong in `tvbo/templates/tvboptim/utils.py`. Import and call these functions in template `<%` blocks.

3. **No Duplicate Logic:** If dfun, cfun, and experiment templates all need the same data (e.g., param names/defaults/shapes), use ONE shared function like `get_param_info()`.

4. **Minimal Unpacking:** Don't unpack metadata into intermediate dicts/lists unless necessary. Prefer direct attribute access: `p.name`, `p.value`, `p.shape`.

5. **Shape Attribute = Array:** If a parameter has a `shape` attribute, it needs array initialization. No need to also check `heterogeneous` - shape is sufficient.

Example - shared utility:
```python
# tvbo/templates/tvboptim/utils.py
def get_param_info(parameters):
    """Extract param names, defaults, and shapes from parameters collection."""
    param_names = [p.name for p in parameters.values()]
    param_defaults = {p.name: float(p.value) if p.value else 1.0 for p in parameters.values()}
    param_shapes = {p.name: str(p.shape) for p in parameters.values() if p.shape}
    return param_names, param_defaults, param_shapes
```

Example - template usage:
```mako
<%
from tvbo.templates.tvboptim.utils import get_param_info
param_names, param_defaults, param_shapes = get_param_info(model.parameters)
%>
```

## Symbolic Mathematics Principles

TVBO aims for a complete symbolic representation of SimulationExperiment using SymPy:

1. **Pure SymPy First:** Always try approaches using built-in SymPy classes (`Sum`, `Product`, `IndexedBase`, `Function`, etc.) before creating custom functions.
2. **No Overriding:** Never override or monkey-patch existing SymPy classes. Extend via subclassing only when absolutely necessary.
3. **Custom Printers Only:** Code generation customization should happen in custom `Printer` subclasses (e.g., `JAXPrinter`), not in the symbolic expressions themselves.
4. **Mathematical Fidelity:** The symbolic expression should match the mathematical notation exactly. E.g., mean = `Sum(...)/N`, not a custom `Mean()` function.
5. **Parseable Strings:** Equation strings in YAML should be parseable by `sympy.parsing.sympy_parser.parse_expr` with a well-defined `local_dict`.

Example - correct approach for aggregated loss:
```python
# Pure SymPy: L = (1/N) * Sum(1 - corr(x[i], y[i]), (i, 0, N-1))
from sympy import Sum, IndexedBase, Symbol, Function
x, y = IndexedBase('x'), IndexedBase('y')
i, N = Symbol('i'), Symbol('N')
corr = Function('correlation')
L = Sum(1 - corr(x[i], y[i]), (i, 0, N-1)) / N  # NOT a custom Mean() function
```

## Core Architecture

- **Data Model:** `schema/tvbo_datamodel.yaml` is the LinkML schema source of truth. Use it for correct metadata handling.
- **Generated Classes:** All major classes inherit from `tvbo.datamodel.schema.py` (auto-generated from LinkML schema).
- **Entry Point:** `tvbo/export/experiment.py` contains `SimulationExperiment`, the main class for running simulations.

## Repository Overview

**TVBO (The Virtual Brain Ontology)** is a Python library for knowledge representation and simulation of large-scale brain network models. It provides:
- Access to TVB ontology and knowledge base for neural mass models
- Dynamical systems definition and simulation with code generation (JAX, NumPy backends)
- YAML-based model/experiment specification using LinkML schema
- Built-in visualization and analysis tools

**Key Stats:** ~20k LOC Python, uses Hatchling build system, requires Python ≥3.11 (CI tests 3.11-3.13).

## Quick Reference Commands

> **CRITICAL — Always use the venv + uv.**
> The project venv lives at `/Users/leonmartin_bih/tools/tvbo/.venv`.
> This project uses **`uv`** as the package manager — always use `uv pip install`,
> never plain `pip install` (plain `pip` is not on PATH in this environment).
> Every terminal command that invokes `python`, `pytest`, `uv`, or `flake8`
> **must** be run inside the activated venv:
> ```bash
> source /Users/leonmartin_bih/tools/tvbo/.venv/bin/activate
> ```
> Never use the system Python or a different interpreter.
> **CI also uses `uv`** via `astral-sh/setup-uv` — use `uv pip install` in CI workflows too.

### Environment Setup (ALWAYS do first)
```bash
cd /Users/leonmartin_bih/tools/tvbo
source .venv/bin/activate            # ALWAYS activate venv first
uv pip install -e .                  # Minimal install
uv pip install -e ".[all]"           # Full install with all extras
uv pip install flake8 pytest pytest-xdist  # Test dependencies
```

### Testing
```bash
source .venv/bin/activate            # ensure venv is active
pytest -x --dist=loadscope           # DEFAULT: fail-fast at first failure
pytest -x --dist=loadscope tests/test_model_loading.py   # Test model YAML loading
pytest -x --dist=loadscope tests/functional/             # Functional tests only
pytest -x --dist=loadscope tests/functional -m backend_jax            # JAX-only backend tests
pytest -x --dist=loadscope tests/functional -m backend_tvb            # TVB-only backend tests
pytest -x --dist=loadscope tests/functional -m backend_pyrates        # PyRates-only backend tests
pytest -x --dist=loadscope tests/functional -m backend_tvboptim       # tvboptim-only backend tests
pytest -x --dist=loadscope tests/functional -m backend_networkdynamics # NetworkDynamics-only tests
```

### Linting (CI uses these exact commands)
```bash
source .venv/bin/activate
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics    # Syntax errors only
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics  # Full lint
```

### Build & Package
```bash
source .venv/bin/activate
python -m build                     # Build sdist and wheel
python -m twine check dist/*        # Verify package metadata
```

### Schema Generation (after modifying schema/tvbo_datamodel.yaml)
```bash
make gen-linkml                     # Regenerate tvbo/datamodel/ from LinkML schema
make gen-openminds                  # Generate openMINDS schemas (optional)
```

## Project Layout

```
tvbo/
├── tvbo/                    # Main Python package
│   ├── __init__.py          # Version (__version__), imports, tempdir setup
│   ├── knowledge/           # Ontology access, query, simulation primitives
│   │   ├── simulation/      # Dynamics, Coupling, Integrator classes
│   │   │   └── localdynamics.py  # Core Dynamics class (~2000 LOC)
│   │   └── ontology.py      # OWL ontology access
│   ├── datamodel/           # AUTO-GENERATED from LinkML (do not edit manually)
│   │   ├── tvbo_datamodel.py    # LinkML dataclasses
│   │   └── tvbopydantic.py      # Pydantic models
│   ├── export/              # Code generation & export
│   │   ├── experiment.py    # SimulationExperiment class (~1900 LOC)
│   │   └── templater.py     # Mako template handling
│   ├── templates/           # Mako code generation templates
│   │   └── autodiff/        # JAX templates (primary backend)
│   ├── api/                 # FastAPI server (optional [api] extra)
│   ├── data/                # Data loading utilities
│   ├── database/            # YAML knowledge base (shipped with package)
│   │   ├── models/          # Neural mass model definitions (*.yaml)
│   │   ├── studies/         # Complete simulation experiment specs
│   │   ├── coupling_functions/  # Coupling function definitions
│   │   ├── integrators/     # Numerical integrator definitions
│   │   └── networks/, atlases/  # Connectome and atlas data
│   └── plot/, analysis/, parse/, utils/
├── schema/                  # LinkML schema source
│   └── tvbo_datamodel.yaml  # Master schema (edit this, run make gen-linkml)
├── tests/                   # Pytest test suite
│   ├── functional/          # Integration tests
│   ├── test_model_loading.py    # Parametrized tests for all database models
│   └── test_tvb_comparison.py   # Comparison with TVB simulator
├── docs/                    # Quarto documentation
├── pyproject.toml           # Build config (hatchling), dependencies, tool config
├── Makefile                 # Build automation targets
└── .github/workflows/       # CI workflows
```

## CI/CD Workflows

### `ci.yml` - Lightweight CI (runs on push/PR to main and dev)

**All triggers:**
1. **Lint** — flake8 syntax errors (blocking) + style warnings (non-blocking)
2. **Compat** — Python 3.11/3.12/3.13 with `pip install -e .` + core tests
3. **Native tests** — `pip install -e ".[all]"` + full pytest (no doc tests, no container)

### `docker.yml` - Docker Build + Container CI (pushes to main/dev, tags)

Pipeline: **build** → **test-container** + **test-docs** (parallel) → **release-ready** (main only)

1. **Build** — Multi-arch Docker image → GHCR + DockerHub
2. **Container tests** — full pytest in the fresh Docker image (no doc tests)
3. **Doc tests** — Quarto + `pytest tests/test_docs.py -m docs` in the fresh Docker image
4. **Release-ready** (main only) — builds sdist + wheel, verifies metadata, uploads artifacts

### `publish-pypi.yml` - PyPI Release (on GitHub release)
1. Tests on Python 3.12, 3.13
2. Builds with `python -m build`
3. Publishes via trusted publishing

**Pre-commit validation:** Always run `flake8 . --select=E9,F63,F7,F82` and `pytest -x --dist=loadscope` before committing.

## Key Classes & Entry Points

```python
from tvbo import Dynamics, SimulationExperiment

# Load model from database
model = Dynamics.from_db("JansenRit")

# Load full experiment
exp = SimulationExperiment.from_db("Schirner2023_MultiscaleBNM_DM")
result = exp.run()
result.plot()

# Generate code
print(exp.render_code('jax'))
```

## Important Conventions

1. **YAML Model Files:** Located in `tvbo/database/models/`. Each file defines a `Dynamics` with `name`, `parameters`, `state_variables`, each with `equation.rhs`.

2. **LinkML Schema:** `schema/tvbo_datamodel.yaml` is the source of truth for data model. After changes, run `make gen-linkml` to regenerate `tvbo/datamodel/`.

3. **Templates:** Code generation uses Mako templates in `tvbo/templates/`. The primary backend is `autodiff/` (JAX).

4. **Tests:** Parametrized tests in `test_model_loading.py` load every YAML model in `tvbo/database/models/`. Add new models there and they'll be tested automatically.

5. **Version:** Defined in `tvbo/__init__.py` as `__version__`. Hatch reads this via `[tool.hatch.version] path = "tvbo/__init__.py"`.

## Known Issues & Workarounds

- **Flake8 recursion error:** Occasionally occurs with sympy in venv; safe to ignore as it doesn't affect CI (sympy is in site-packages, not project code).
- **Some tests fail on dev branch:** ~10 tests currently fail due to ongoing development. Focus on `tests/functional/` and `test_model_loading.py` for core validation.
- **Optional extras:** `[tvb]` extra requires TVB packages. `[knowledge]` requires manual git install of neurommsig-knowledge.

## Validation Checklist

Before submitting changes (always activate the venv first: `source .venv/bin/activate`):
1. ✅ `uv pip install -e .` succeeds
2. ✅ `flake8 . --select=E9,F63,F7,F82` returns 0 (no syntax errors)
3. ✅ `pytest -x --dist=loadscope tests/functional/` passes
4. ✅ If schema changed: `make gen-linkml` and commit generated files
5. ✅ `python -c "from tvbo import Dynamics, SimulationExperiment"` succeeds

## File Types Reference

| Extension | Purpose | Location |
|-----------|---------|----------|
| `.yaml` | Model/experiment definitions | `tvbo/database/` |
| `.py.mako` | Code generation templates | `tvbo/templates/` |
| `.owl` | OWL ontology (read-only) | accessed via owlready2 |
| `.qmd` | Quarto documentation | `docs/` |

Trust these instructions. Only search the codebase if information here is incomplete or found to be incorrect.

<!-- mermaid-ai-skills:start -->
## Mermaid Diagrams

When the user asks to create, edit, or visualize a diagram, follow the
instructions in `.github/instructions/mermaid.instructions.md`.
<!-- mermaid-ai-skills:end -->
