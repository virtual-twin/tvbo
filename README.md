<a href="imgs/tvbo_logo.png">
    <img src="imgs/tvbo_logo.png" alt="TVBO logo" title="TVBO" align="right" height="100" />
</a>

# The Virtual Brain Ontology

[![Python package](https://github.com/the-virtual-brain/tvb-o/workflows/Python%20package/badge.svg)](https://github.com/the-virtual-brain/tvb-o/actions?query=workflow%3A%22Python+package%22)
[![PyPI version](https://img.shields.io/pypi/v/tvbo.svg)](https://pypi.org/project/tvbo/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/tvbo.svg)](https://pypi.org/project/tvbo/)

`tvbo` is a Python library to access the knowledge representation system (i.e., ontology) and data model for the neuroinformatics platform The Virtual Brain (TVB).

## 🚀 Installation

```bash
pip install tvbo
```

## 📖 Quick Start

### Example: Lorenz Attractor Simulation

<details>
<summary><b>📝 Model Specification (YAML)</b></summary>

```yaml
name: LorenzAttractor
parameters:
    sigma:
        value: 10
        label: Prandtl number
    rho:
        label: Rayleigh number
        value: 28
    beta:
        value: 2.6666666666666665
state_variables:
    X:
        equation:
            lhs: \dot{X}
            rhs: sigma * (Y - X)
    Y:
        equation:
            lhs: \dot{Y}
            rhs: X * (rho - Z) - Y
    Z:
        equation:
            lhs: \dot{Z}
            rhs: X * Y - beta * Z
```

</details>

<details>
<summary><b>🔧 Generate Code</b></summary>

```python
from tvbo import Dynamics, SimulationExperiment

lorenz = Dynamics(
    parameters={
        "sigma": {"value": 10.0},
        "rho": {"value": 28.0},
        "beta": {"value": 8 / 3},
    },
    state_variables={
        "X": {"equation": {"rhs": "sigma * (Y - X)"}},
        "Y": {"equation": {"rhs": "X * (rho - Z) - Y"}},
        "Z": {"equation": {"rhs": "X * Y - beta * Z"}},
    },
)

code = SimulationExperiment(local_dynamics=lorenz).render_code('jax')
print(code)
```

</details>

<details>
<summary><b>▶️ Run Simulation</b></summary>

```python
from tvbo import Dynamics, SimulationExperiment

lorenz = Dynamics(
    parameters={
        "sigma": {"value": 10.0},
        "rho": {"value": 28.0},
        "beta": {"value": 8 / 3},
    },
    state_variables={
        "X": {"equation": {"rhs": "sigma * (Y - X)"}},
        "Y": {"equation": {"rhs": "X * (rho - Z) - Y"}},
        "Z": {"equation": {"rhs": "X * Y - beta * Z"}},
    },
)

# Run simulation and plot results
SimulationExperiment(local_dynamics=lorenz).run(duration=1000).plot()
```

</details>

## 📚 Documentation

- **[Full Documentation](https://virtual-twin.github.io/tvbo/)**
- **[Model Browser](https://virtual-twin.github.io/tvbo/browser)** - Browse models, parameters, and equations
- **[Metadata Schema](https://virtual-twin.github.io/tvbo/datamodel)** - Explore the TVB-O data model

## 🔬 Features

- 🧠 Access TVB ontology and knowledge base
- 📊 Define and simulate dynamical systems
- 🔄 Code generation for multiple backends (JAX, NumPy)
- 📈 Built-in visualization tools
- 🗃️ Structured metadata schema

## 📦 Installation Options

### Standard Installation
```bash
pip install tvbo
```

### With API Server Support
```bash
pip install tvbo[api]
```

### With TVB Integration
```bash
pip install tvbo[tvb]
```

### Full Installation (All Features)
```bash
pip install tvbo[all]
```

> **Note:** The `knowledge` extra requires manual installation:
> ```bash
> pip install git+https://github.com/neurommsig/neurommsig-knowledge.git
> ```

## 📄 License

Copyright © 2025 Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.
