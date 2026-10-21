#
# Module: classes/__init__.py
#
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""tvbo.classes.

Core simulation classes: dynamics, coupling, noise, continuation, observation, perturbation, equation, functions, experiment, study, network, and atlas.
"""

_LAZY_IMPORTS = {
    "Atlas": ("tvbo.classes.atlas", "Atlas"),
    "Continuation": ("tvbo.classes.continuation", "Continuation"),
    "Coupling": ("tvbo.classes.coupling", "Coupling"),
    "Dynamics": ("tvbo.classes.dynamics", "Dynamics"),
    "Function": ("tvbo.classes.function", "Function"),
    "LossFunction": ("tvbo.classes.function", "LossFunction"),
    "Model": ("tvbo.classes.dynamics", "Dynamics"),
    "Network": ("tvbo.classes.network", "Network"),
    "Noise": ("tvbo.classes.noise", "Noise"),
    "SimulationExperiment": ("tvbo.classes.experiment", "SimulationExperiment"),
    "SimulationStudy": ("tvbo.classes.study", "SimulationStudy"),
    "SimulationTool": ("tvbo.classes.software", "SimulationTool"),
}
"""Each public name, as the module that defines it and the name it has there, imported on first access."""

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'tvbo.classes' has no attribute {name!r}")
    import importlib

    module, attr = _LAZY_IMPORTS[name]
    value = getattr(importlib.import_module(module), attr)
    globals()[name] = value
    return value
