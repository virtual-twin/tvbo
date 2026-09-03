# Developer Guide

## Git Workflow

TVBO uses a standard **GitHub Flow** with a two-branch model:

- **`dev`**: active development branch
- **`main`**: stable release branch, protected by CI

### Contributing

This follows standard open-source contribution practice:

1. **Fork or branch** — external contributors fork the repo; team members create feature branches off `dev`.
2. **Create a feature branch** — name it descriptively (e.g., `feat/jax-backend`, `fix/import-error`, `docs/add-tutorial`).
3. **Develop and commit** — make focused commits with clear messages.
4. **Open a Pull Request** — PR your feature branch into `dev` (for ongoing work) or `dev` → `main` (for releases).
5. **CI runs automatically** — all checks must pass.
6. **Code review** — at least one approving review before merge.
7. **Merge** — use squash-merge or merge commit, then delete the feature branch.

### Branch naming conventions

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat/` | New feature | `feat/symbolic-export` |
| `fix/` | Bug fix | `fix/optional-imports` |
| `docs/` | Documentation only | `docs/developer-guide` |
| `refactor/` | Code restructure, no behavior change | `refactor/template-utils` |
| `ci/` | CI/CD changes | `ci/update-actions-v6` |
| `test/` | Adding or fixing tests | `test/coupling-functions` |

### Typical workflow

```bash
# 1. Start from dev
git checkout dev
git pull origin dev

# 2. Create feature branch
git checkout -b feat/my-feature

# 3. Work, commit
git add -A
git commit -m "Add new feature X"

# 4. Push and open PR
git push -u origin feat/my-feature
# → Open PR on GitHub: feat/my-feature → dev
```

For release merges (`dev` → `main`), the same PR process applies — CI gates protect `main`.

---

## CI/CD Workflows

TVBO has **5 GitHub Actions workflows**. Here is what each one does, when it runs, and what it produces.

### 1. CI (`ci.yml`)

> **Runs on:** push to `main` or `dev`, pull requests targeting `main`

The core quality gate. Ensures code is correct before it reaches `main`.

```
┌─────────────────────────────────────────────────┐
│                  CI Pipeline                     │
│                                                  │
│  ┌──────┐                                        │
│  │ Lint │ ruff syntax check (blocking)           │
│  └──┬───┘ ruff lint/format + mypy (non-blocking) │
│     │                                            │
│     ├──────────────┐                             │
│     ▼              ▼                             │
│  ┌──────┐    ┌───────────┐                       │
│  │ Test │    │ Test Docs │                       │
│  └──┬───┘    └─────┬─────┘                       │
│     │  pytest on   │  Execute all                │
│     │  Python      │  .qmd notebooks             │
│     │  3.10–3.13   │                             │
│     └──────┬───────┘                             │
│            ▼                                     │
│     ┌───────────┐                                │
│     │  Package  │  (main branch only)            │
│     └───────────┘                                │
│     Build sdist + wheel → upload artifact        │
│                                                  │
│  ❌ Any failure in Lint/Test/Test Docs            │
│     → PR cannot be merged                        │
└─────────────────────────────────────────────────┘
```

| Job | What it does | Blocks merge? |
|-----|-------------|---------------|
| **Lint** | `ruff` syntax errors (`E9,F63,F7,F82`, blocking) + full lint + `ruff format --check` + `mypy` (non-blocking) | ✅ Yes (syntax errors) |
| **Test** | `pytest -q` on Python 3.10, 3.11, 3.12, 3.13 (matrix) | ✅ Yes |
| **Test Docs** | Converts all `.qmd` doc pages to notebooks and executes them end-to-end | ✅ Yes |
| **Package** | Builds sdist + wheel, uploads as artifact (only on `main` push) | N/A |

**All three gate jobs must pass before a PR can be merged.**

### 2. Docs (`docs-deploy.yml`)

> **Runs on:** called by `docker.yml` once the release image is pushed, or dispatched manually against any tag

Publishes the [Quarto](https://quarto.org/) documentation site to GitHub Pages. It is release-driven rather than branch-driven: the render happens *inside* the released container, at that image's locked dependency set, so the site is a faithful cold re-execution of every page rather than a replay of cached output.

| Step | Details |
|------|---------|
| Render | Inside the released image, with `_freeze`, `.quarto` and the Actions cache all dropped |
| Build ontology | `make gen-merged gen-shacl` in an isolated venv, so the published spec is generated rather than read from the repository |
| Widoco spec | Runs on the host, overlaying the W3C-style HTML and WebVOWL onto the built `_site` |
| Publish | `quarto publish gh-pages --no-render`, reusing the `_site` just built |

- A dispatch defaults to *not* publishing, so the cold build can be exercised without deploying.
- Requires `contents: write` permission to push to the `gh-pages` branch.

### 3. Docker (`docker.yml`)

> **Runs on:** push to `main` or `dev`, version tags (`v*`), manual dispatch

Builds multi-platform Docker images and pushes to two registries.

| Detail | Value |
|--------|-------|
| **Platforms** | `linux/amd64`, `linux/arm64` |
| **Registries** | Docker Hub (`leonmartin2/tvbo`) + GHCR (`ghcr.io/virtual-twin/tvbo`) |
| **Cache** | GitHub Actions cache (`type=gha`) |

**Image tags depend on the trigger:**

| Trigger | Tags produced |
|---------|---------------|
| Push to `main` | `main`, `latest`, `<commit-sha>` |
| Push to `dev` | `dev`, `<commit-sha>` |
| Version tag (e.g., `v0.5.0`) | `0.5.0`, `<commit-sha>` |

### 4. Publish to PyPI (`publish-pypi.yml`)

> **Runs on:** GitHub Release published, manual dispatch

The full release-to-PyPI pipeline with its own test gate.

```
┌───────────────────────────────────────┐
│        PyPI Publish Pipeline          │
│                                       │
│  1. Test (Python 3.12, 3.13)          │
│     └─► pytest -q                     │
│                                       │
│  2. Build                             │
│     └─► python -m build               │
│     └─► twine check dist/*            │
│     └─► Upload artifact               │
│                                       │
│  3. Publish                           │
│     └─► Download artifact             │
│     └─► pypa/gh-action-pypi-publish   │
│         (OIDC Trusted Publishing)     │
└───────────────────────────────────────┘
```

- Uses **Trusted Publishing** (OIDC) — no API token needed, GitHub identity is verified by PyPI directly.
- The `id-token: write` permission is required for OIDC.
- Tests run again independently (not reusing CI results) to ensure the release commit is clean.

### 5. Ontology (the `ontology` job in `publish-pypi.yml`)

> **Runs on:** every run of `publish-pypi.yml`, and attaches to the release when there is one

The ontology is built from the released schema and shipped with the package that determines it, so there is one tag and one version rather than a parallel `ontology-v*` series that could name a different state of the schema. It runs `build` after it, not beside it, because the wheel carries a copy of the merged ontology at `tvbo/data/ontology/tvbo.owl` — building the two in parallel is how a release ends up with a fresh ontology attached and a stale one inside the wheel.

| Step | Details |
|------|---------|
| Build | `make gen-merged gen-shacl` — needs ROBOT and a JVM, which is why this is not a per-commit gate |
| Version | `gen-merged` stamps the version IRI from `__version__`, the same file hatch reads the wheel version from |
| Hand to `build` | The merged ontology goes to the build job as an artifact, which overwrites the committed copy before `python -m build` |
| Attach | On a release only: `sha256sum` plus `gh release upload <tag>` with `tvbo.owl`, `tvb-o-struct.owl`, `tvb-o.shacl.ttl` and the checksum file |

- `https://w3id.org/tvbo/<version>/tvbo.owl` resolves to the same version you can `pip install`, and to the copy inside it.
- `tvb-o-struct.owl` and `tvb-o.shacl.ttl` are not committed; the release is where a versioned copy comes from, and `schema-artifacts` uploads them per-run for anything that needs the current tip.
- A manual dispatch builds the ontology without attaching anything, so a break shows up while it is still cheap to fix rather than during a release.

---

## What Happens When…

### …you open a Pull Request to `main`

The **CI** workflow triggers. All three gate jobs (Lint, Test, Test Docs) must pass. The Package job does **not** run on PRs — only on merge.

### …a PR is merged to `main`

Three things happen simultaneously:

1. **CI** runs again (lint + test + test-docs + **package build**)
2. **Docker** builds and pushes images tagged `main`, `latest`, `<sha>`
3. Nothing happens on PyPI, to the docs site or to the ontology — all three wait for a GitHub Release

### …you push to `dev`

Only **CI** (lint + test + test-docs, no package) and **Docker** (tagged `dev`, `<sha>`) run. Docs are not published.

### …you create a GitHub Release

**Publish to PyPI** runs: tests → builds → publishes the package to PyPI via Trusted Publishing.

### …you trigger a workflow manually

CI, Docs, Docker, and PyPI workflows all support `workflow_dispatch` — you can run them from the GitHub Actions tab at any time.

---

## Release Process

1. Develop on `dev` (or feature branches → `dev`)
2. Open PR: `dev` → `main`
3. CI validates everything (lint, tests, doc notebooks)
4. Get code review approval, then merge
5. Merge triggers: Docker build, docs publish, package artifact, ontology release (if OWL changed)
6. **To publish to PyPI:** create a GitHub Release with a version tag (e.g., `v0.5.0`)
   - First update `__version__` in `tvbo/__init__.py`
   - Merge that change to `main`
   - Create a release on GitHub → PyPI publish workflow runs automatically

---

## Local Development

### Setup

```bash
git clone https://github.com/virtual-twin/tvbo.git
cd tvbo
git checkout dev
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

The `[all]` extra installs `pycobi`, but the AUTO-07p continuation engine it wraps is a native Fortran build that pip cannot place. Provision it once — the link lives in the venv, so re-run this after recreating `.venv`:

```bash
tvbo install auto7p            # links an existing $AUTO_DIR install
tvbo install auto7p --build    # or builds AUTO-07p from source first
```

Without it, the bifurcation-continuation pages and PyRates continuation examples fail with `ModuleNotFoundError: No module named 'auto'`. See [Installation → AUTO-07p continuation](https://virtual-twin.github.io/tvbo/installation.html#auto-07p-continuation-auto7p)
for the manual equivalent.

### Running Tests

```bash
# All tests (excluding docs)
pytest -q

# Specific test file
pytest tests/test_model_loading.py -v

# Doc notebook tests (requires Quarto)
pytest tests/test_docs.py -v -m docs

# Functional tests only
pytest tests/functional/ -v
```

### Linting & Formatting

We use [`ruff`](https://docs.astral.sh/ruff/) for linting + formatting and [`mypy`](https://mypy.readthedocs.io/) for type checks. Install via `uv pip install ruff mypy` or use the `dev` env.

```bash
# Blocking errors (must pass before merge): same gate as CI
ruff check --select=E9,F63,F7,F82 .

# Full lint summary (non-blocking)
ruff check --statistics .

# Auto-fix what can be fixed
ruff check --fix .

# Format check / apply
ruff format --check .   # report only
ruff format .           # apply

# Type check (non-blocking, ratcheting toward zero)
mypy tvbo
```

### Building Docs Locally

```bash
cd docs
quarto preview   # Live preview at localhost:4000
quarto render    # Full build to docs/_site/
```

### Building the Package

```bash
pip install build
python -m build          # Creates dist/tvbo-*.tar.gz and dist/tvbo-*.whl
python -m twine check dist/*  # Verify package metadata
```

---

## Project Structure

```
.github/workflows/
├── ci.yml               # Lint + Test + Test Docs + Package
├── docker.yml           # Docker multi-platform build & push
├── docs-deploy.yml      # Quarto publish to GitHub Pages
└── publish-pypi.yml     # PyPI release + ontology artifacts (on GitHub Release)
```

## Schema Changes

If you modify `schema/tvbo_datamodel.yaml`:

```bash
make gen-linkml          # Regenerate tvbo/datamodel/
```

Always commit the regenerated files alongside schema changes.

## Validation Checklist

Before opening a PR:

- [ ] `pip install -e .` succeeds
- [ ] `ruff check --select=E9,F63,F7,F82 .` returns 0
- [ ] `pytest -q --ignore=tests/test_docs.py` passes
- [ ] `python -c "from tvbo import Dynamics, SimulationExperiment"` works
- [ ] If schema changed: `make gen-linkml` and commit generated files
