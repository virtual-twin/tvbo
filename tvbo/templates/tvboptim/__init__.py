#
# Module: __init__.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
# TVB-Optim Templates

Mako templates for generating tvboptim network dynamics code from TVBO models.

## Templates

- `tvbo-tvboptim-dfun.py.mako`: Dynamics class generation.
- `tvbo-tvboptim-cfun.py.mako`: Coupling function generation.
- `tvbo-tvboptim-solver.py.mako`: Solver/integrator generation.
- `tvbo-tvboptim-noise.py.mako`: Noise model generation.
- `tvbo-tvboptim-observation.py.mako`: Observation/monitor generation (metadata-driven pipelines).
- `tvbo-tvboptim-optim.py.mako`: Optimizer configuration generation.
- `tvbo-tvboptim-exploration.py.mako`: Parameter exploration (grid search) generation.
- `tvbo-tvboptim-sim.py.mako`: Full simulation workflow generation.
- `tvbo-tvboptim-experiment.py.mako`: Complete experiment generation (fully metadata-driven).

## Usage

From a TVBO `SimulationExperiment`:
```python
from tvbo import SimulationExperiment
from tvbo.templates import lookup

experiment = SimulationExperiment.from_file("my_experiment.yaml")

# Generate full workflow
template = lookup.get_template("tvboptim/tvbo-tvboptim-sim.py.mako")
code = template.render(experiment=experiment)

# Generate individual components
dfun_template = lookup.get_template("tvboptim/tvbo-tvboptim-dfun.py.mako")
dfun_code = dfun_template.render(model=experiment.dynamics)

# Generate exploration workflow
expl_template = lookup.get_template("tvboptim/tvbo-tvboptim-exploration.py.mako")
expl_code = expl_template.render(experiment=experiment)

# Generate optimization workflow
optim_template = lookup.get_template("tvboptim/tvbo-tvboptim-optim.py.mako")
optim_code = optim_template.render(experiment=experiment)
```
## Template Context Variables:

All templates accept:
- experiment: SimulationExperiment instance

Individual component templates also accept:
- model: Dynamics instance (for dfun)
- coupling: Coupling instance (for cfun)
- integration: Integration instance (for solver, noise)
- monitors: List of Monitor instances (for observation)
- fitting: ModelFitting instance (for target, loss, optim)

## Output Format:

Templates generate Python code compatible with tvboptim.experimental.network_dynamics

which provides:
- JAX-based differentiable simulation
- Support for both instantaneous and delayed coupling
- Stochastic (SDE) and deterministic (ODE) modes
- BOLD signal monitoring and FC computation
- Integration with optax for parameter optimization
- Grid exploration and parallel execution
"""


# Template names for programmatic access
TEMPLATES = {
    'dfun': 'tvboptim/tvbo-tvboptim-dfun.py.mako',
    'cfun': 'tvboptim/tvbo-tvboptim-cfun.py.mako',
    'solver': 'tvboptim/tvbo-tvboptim-solver.py.mako',
    'noise': 'tvboptim/tvbo-tvboptim-noise.py.mako',
    'observation': 'tvboptim/tvbo-tvboptim-observation.py.mako',
    'optim': 'tvboptim/tvbo-tvboptim-optim.py.mako',
    'exploration': 'tvboptim/tvbo-tvboptim-exploration.py.mako',
    'sim': 'tvboptim/tvbo-tvboptim-sim.py.mako',
    'experiment': 'tvboptim/tvbo-tvboptim-experiment.py.mako',
}

__all__ = [
    "TEMPLATES",
]
