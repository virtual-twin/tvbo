---
name: tests-and-backends
description: pytest marker conventions for backend-tagged tests in TVBO, the slow/julia
  markers, and how to run the test suite locally.
---

# Tests and Backends

TVBO's test suite is **backend-tagged**: every backend integration test carries a marker so CI can run a subset depending on which extras are installed.

## Registered markers (`[tool.pytest.ini_options].markers`)

| Marker | Meaning |
|--------|---------|
| `backend_core` | Backend-agnostic core simulation tests |
| `backend_jax` | JAX backend |
| `backend_tvb` | TVB backend |
| `backend_pyrates` | PyRates backend |
| `backend_tvboptim` | tvboptim backend |
| `backend_julia` | Class-level Julia tests (any Julia-backed sim) |
| `backend_julia_diffeq` | DifferentialEquations.jl |
| `backend_networkdynamics` | NetworkDynamics.jl |
| `backend_mtk` | ModelingToolkit.jl |
| `backend_neuroml` | NeuroML/LEMS |
| `docs` | Quarto/Jupyter notebook execution tests |
| `julia` | Needs a Julia runtime via `juliacall` |
| `slow` | Long-running; excluded by default — pass `--run-slow` to include |

When adding a backend test, **always** apply the matching `backend_*` marker. CI uses these to filter.

## Running tests

```bash
# Fast, default subset
pytest

# Include slow tests
pytest --run-slow

# A specific backend
pytest -m backend_jax

# Exclude Julia entirely
pytest -m "not julia"
```

## Test layout

- `tests/` — the canonical suite.
- `dev/test/` — exploratory scripts and notebooks. **Not** picked up by `pytest`.

## Pytest config

- Default options: `-n 1 --dist=loadscope` (single-process, group-by-file). Don't override globally unless you have a reason.
- New markers must be **registered** in `pyproject.toml` or pytest will warn (`--strict-markers`).

## Adding a new backend marker

Edit `[tool.pytest.ini_options].markers` in `pyproject.toml`. Add the optional-dependency group under `[project.optional-dependencies]` so users can `pip install tvbo[<backend>]`. See the `codegen-templates` skill for the full new-backend checklist.
