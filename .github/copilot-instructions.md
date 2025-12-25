# Copilot Coding Agent Instructions for TVBO

## Code Style Principles

1. **Minimal & Clean:** Write the shortest code that solves the problem. No boilerplate.
2. **No Redundancy:** Always check if logic already exists before defining new functions. Reuse existing code.
3. **MVP First:** No fallbacks, no try-except blocks. Code should work as expected; if it breaks, we debug.
4. **Readable:** Clear variable names, simple control flow, no unnecessary abstractions.

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
