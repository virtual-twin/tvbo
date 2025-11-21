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
