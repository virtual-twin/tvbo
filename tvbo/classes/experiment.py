"""The simulation experiment class for TVBO.

[`SimulationExperiment`](#tvbo.classes.experiment.SimulationExperiment) is the generated class with its behaviour (`tvbo.behaviour.experiment.SimulationExperimentBehaviour`): it binds local dynamics, a network, coupling, integration settings, monitors and stimulation into a single, YAML-round-trippable object, and is the entry point for constructing an experiment (from YAML, a file, the platform, or a TVB simulator), configuring it, rendering backend code, and running it on any of the supported backends.
"""

from tvbo import datamodel as tvbo_datamodel
from tvbo import templates

__all__ = ["SimulationExperiment", "templates"]

SimulationExperiment = tvbo_datamodel.SimulationExperiment
"""The generated `SimulationExperiment`, which carries its behaviour itself; this is its public import location."""
