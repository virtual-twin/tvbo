# Copilot Coding Agent Instructions for TVBO

## Code Organization Principles

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

## Declarative Data Access Principles

**Data access should mirror YAML schema structure.** Users access results using the same path as the YAML definition:

1. **Schema-Aligned Access:** Result structure mirrors YAML sections. E.g., `results.integration.main`, `results.algorithms.fic`, `results.optimization.loss_fc`.
2. **No Implementation Details:** Never expose internal mechanisms via underscore prefixes (e.g., `._raw`). Users access `.data`, `.observations`, not internal representations.
3. **Wrap at Boundaries:** Internal functions use raw data types; only final user-facing returns wrap in result classes (`SimulationResult`, `AlgorithmResult`, etc.).
4. **Consistent Nesting:** `results.integration.main.observations.bold` - observations attached to the simulation that produced them.
5. **Convenience Aliases:** For common access patterns, provide both nested and flat access: `results.algorithms.fic` and `results.fic`.

Example - YAML to Python access:
```yaml
# YAML
integration:
  main:
    duration: 600000
algorithms:
  fic:
    n_iterations: 100
```

```python
# Python - mirrors YAML structure
results.integration.main.data        # Simulation data
results.integration.main.observations.bold  # BOLD from main simulation
results.algorithms.fic.state         # FIC tuned state
results.algorithms.fic.history       # Per-iteration tracking
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
- **Generated Classes:** All major classes inherit from `tvbo/datamodel/tvbo_datamodel.py` (auto-generated from LinkML schema).
- **Entry Point:** `tvbo/export/experiment.py` contains `SimulationExperiment`, the main class for running simulations.

## Repository Overview

**TVBO (The Virtual Brain Ontology)** is a Python library for knowledge representation and simulation of large-scale brain network models. It provides:
- Access to TVB ontology and knowledge base for neural mass models
- Dynamical systems definition and simulation with code generation (JAX, NumPy backends)
- YAML-based model/experiment specification using LinkML schema
- Built-in visualization and analysis tools

**Key Stats:** ~20k LOC Python, uses Hatchling build system, requires Python ≥3.10 (CI tests 3.10-3.13).

## Quick Reference Commands

### Environment Setup (ALWAYS do first)
```bash
cd /path/to/tvbo
python -m venv .venv
source .venv/bin/activate
pip install -e .                    # Minimal install
pip install -e ".[all]"             # Full install with all extras
pip install flake8 pytest           # Test dependencies
```

### Testing
```bash
pytest -q                           # Run all tests (~22s)
pytest tests/test_model_loading.py  # Test model YAML loading
pytest tests/functional/            # Functional tests only
```

### Linting (CI uses these exact commands)
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics    # Syntax errors only
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics  # Full lint
```

### Build & Package
```bash
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
│   └── plot/, analysis/, parse/, utils/
├── database/                # YAML knowledge base
│   ├── models/              # Neural mass model definitions (*.yaml)
│   ├── studies/             # Complete simulation experiment specs
│   ├── coupling_functions/  # Coupling function definitions
│   ├── integrators/         # Numerical integrator definitions
│   └── networks/, atlases/  # Connectome and atlas data
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

### `ci.yml` - Python Package (runs on push/PR to main)
1. Tests on Python 3.10, 3.11, 3.12
2. Installs: `pip install flake8 pytest nbformat && pip install .[all]`
3. Runs flake8 lint (syntax errors are blocking)
4. Runs `pytest -q`

### `publish-pypi.yml` - PyPI Release (on GitHub release)
1. Tests on Python 3.12, 3.13
2. Builds with `python -m build`
3. Publishes via trusted publishing

**Pre-commit validation:** Always run `flake8 . --select=E9,F63,F7,F82` and `pytest -q` before committing.

## Key Classes & Entry Points

```python
from tvbo import Dynamics, SimulationExperiment

# Load model from YAML
model = Dynamics.from_file("database/models/JansenRit.yaml")

# Load full experiment
exp = SimulationExperiment.from_file("database/studies/Schirner2023.yaml")
result = exp.run()
result.plot()

# Generate code
print(exp.render_code('jax'))
```

## Important Conventions

1. **YAML Model Files:** Located in `database/models/`. Each file defines a `Dynamics` with `name`, `parameters`, `state_variables`, each with `equation.rhs`.

2. **LinkML Schema:** `schema/tvbo_datamodel.yaml` is the source of truth for data model. After changes, run `make gen-linkml` to regenerate `tvbo/datamodel/`.

3. **Templates:** Code generation uses Mako templates in `tvbo/templates/`. The primary backend is `autodiff/` (JAX).

4. **Tests:** Parametrized tests in `test_model_loading.py` load every YAML model in `database/models/`. Add new models there and they'll be tested automatically.

5. **Version:** Defined in `tvbo/__init__.py` as `__version__`. Hatch reads this via `[tool.hatch.version] path = "tvbo/__init__.py"`.

## Known Issues & Workarounds

- **Flake8 recursion error:** Occasionally occurs with sympy in venv; safe to ignore as it doesn't affect CI (sympy is in site-packages, not project code).
- **Some tests fail on dev branch:** ~10 tests currently fail due to ongoing development. Focus on `tests/functional/` and `test_model_loading.py` for core validation.
- **Optional extras:** `[tvb]` extra requires TVB packages. `[knowledge]` requires manual git install of neurommsig-knowledge.

## Validation Checklist

Before submitting changes:
1. ✅ `pip install -e .` succeeds
2. ✅ `flake8 . --select=E9,F63,F7,F82` returns 0 (no syntax errors)
3. ✅ `pytest -q tests/functional/` passes
4. ✅ If schema changed: `make gen-linkml` and commit generated files
5. ✅ `python -c "from tvbo import Dynamics, SimulationExperiment"` succeeds

## File Types Reference

| Extension | Purpose | Location |
|-----------|---------|----------|
| `.yaml` | Model/experiment definitions | `database/` |
| `.py.mako` | Code generation templates | `tvbo/templates/` |
| `.owl` | OWL ontology (read-only) | accessed via owlready2 |
| `.qmd` | Quarto documentation | `docs/` |

Trust these instructions. Only search the codebase if information here is incomplete or found to be incorrect.
