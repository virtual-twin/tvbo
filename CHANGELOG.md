## 0.2.1 – 2025-11-20
### Changed
- Dependency/test adjustments: skip heavy `tvb` & notebook tests in release workflow; optional deps not required for minimal install.

### Fixed
- Minor packaging workflow refinements preparing for PyPI publish (trusted publishing test exclusions).

### Notes
- Incremental release before broader 1.0.0 stabilization; semantic versioning pre-1.0: minor component changes still bump patch.

## 1.0.0 – 2025-11-12
### Added
- First stable release of the `tvbo` Python package (The Virtual Brain Ontology).
- Ontology-backed simulation components:
  - Dynamics construction from ontology classes (e.g., Jansen–Rit) with parameters/state variables.
  - `SimulationExperiment` assembly with default coupling, integration, and network components.
  - `TimeSeries` utilities for extracting state variables and basic frequency analysis.
- Optional installation extras to keep base installs lean:
  - `api` (FastAPI + uvicorn)
  - `tvb` (TVB framework + library)
  - `knowledge` (neurommsig knowledge integration)
- CI workflow: unified "Python package" (multi-version tests + lint + fast functional tests + artifacts).
- Release workflow: version/tag verification, changelog section enforcement, sdist/wheel build, SHA256 checksums, automatic release note population, PyPI Trusted Publishing.

### Changed
- Replaced heavy notebook execution in CI with lightweight functional smoke tests inspired by notebooks for speed and stability.

### Removed
- Legacy notebook execution test strategy.

### Fixed
- Robust access to model parameter/state variable containers across varying representations.
