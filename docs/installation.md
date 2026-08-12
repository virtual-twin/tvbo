# Installation

## Requirements

- Python 3.12 or later
- pip (Python package manager)

## Install via pip

Install the latest release from PyPI:

```bash
pip install tvbo
```

## Install from GitHub

Install the development version directly from GitHub:

```bash
pip install git+https://github.com/virtual-twin/tvbo.git
```

For a specific branch:

```bash
pip install git+https://github.com/virtual-twin/tvbo.git@main
```

## Install from source

Clone the repository and install in development mode:

```bash
git clone https://github.com/virtual-twin/tvbo.git
cd tvbo
pip install -e .
```

## Optional features (extras)

Optional feature sets install as pip extras. Request one or several in brackets:

```bash
pip install "tvbo[tvboptim]"        # JAX-based simulation & optimization
pip install "tvbo[pyrates]"         # PyRates modeling + AUTO-07p continuation
pip install "tvbo[all]"             # every backend and optional feature
```

Most extras are self-contained. Two need a component that is not on PyPI and
therefore cannot be installed by `pip` alone — they are documented below.

### AUTO-07p continuation (`auto7p`)

Numerical bifurcation continuation runs on [AUTO-07p](https://github.com/auto-07p/auto-07p),
a Fortran engine, through the [pycobi](https://pypi.org/project/pycobi/) wrapper.

```bash
pip install "tvbo[auto7p]"     # the pip half: the pycobi wrapper
tvbo install auto7p            # the native half: locate + link AUTO-07p
```

`pip` installs **pycobi** (the Python wrapper). The AUTO-07p **engine** is not
distributed on PyPI, so `tvbo install auto7p` provisions it: it locates an
existing build — via `--auto-dir`, `$AUTO_DIR`, or common locations — and links
its Python front-end into the active environment. If no build is found, add
`--build` to clone and compile AUTO-07p from source first (needs `git`, `make`,
and a Fortran compiler):

```bash
tvbo install auto7p --auto-dir /opt/auto-07p   # use a specific install
tvbo install auto7p --build                    # build from source, then link
```

> **Re-run `tvbo install auto7p` after recreating the virtualenv.** The link
> lives inside the venv (usually gitignored and rebuilt from `pyproject.toml`),
> and because AUTO-07p is not a declared dependency it is not restored
> automatically.

Verify the engine is reachable:

```python
import pycobi, auto  # both import cleanly once AUTO-07p is linked

print("AUTO-07p continuation ready")
```

**What the command does.** AUTO-07p ships its Python module under
`$AUTO_DIR/python`, but nothing puts that on the import path — neither `pip`, nor
AUTO's own `auto.env.sh`, which sets `AUTO_DIR` and `PATH` but not `PYTHONPATH`.
`tvbo install auto7p` writes a one-line `.pth` link into the environment's
site-packages, equivalent to:

```bash
echo "$AUTO_DIR/python" \
  > "$(python -c 'import site; print(site.getsitepackages()[0])')/auto-07p.pth"
```

For continuation *runs* (not just imports), also `export AUTO_DIR=/path/to/auto-07p`
in your shell profile so pycobi can find AUTO's command-line tools.

## Docker

Run TVBO in a containerized environment with all dependencies pre-installed.

### API Server

Run the TVBO API server:

```bash
docker run -p 8000:8000 -e MODE=api tvbo:latest
```

Access the API at `http://localhost:8000`

### Jupyter Lab

Run TVBO with Jupyter Lab:

```bash
docker run -p 8888:8888 -e MODE=jupyter tvbo:latest
```

Access Jupyter Lab at `http://localhost:8888`

### Build from Dockerfile

Build the Docker image locally:

```bash
docker build -t tvbo:latest .
```

## Verify Installation

Test your installation:

```python
from tvbo import Dynamics

print("TVBO successfully installed!")
```
