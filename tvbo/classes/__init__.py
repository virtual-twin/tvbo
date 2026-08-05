#
# Module: classes/__init__.py
#
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""
tvbo.classes
============
Core simulation classes: dynamics, coupling, noise, continuation, observation,
perturbation, equation, functions, experiment, study, network, and atlas.
"""

__all__ = [
    "Atlas",
    "Connectome",
    "Continuation",
    "Coupling",
    "Dynamics",
    "Function",
    "LossFunction",
    "Model",
    "Network",
    "Noise",
    "SimulationExperiment",
    "SimulationStudy",
    "StudyCollection",
]


def __getattr__(name):
    if name in ("Dynamics", "Model"):
        from tvbo.classes.dynamics import Dynamics

        globals()["Dynamics"] = Dynamics
        globals()["Model"] = Dynamics
        return Dynamics
    if name == "Coupling":
        from tvbo.classes.coupling import Coupling

        globals()["Coupling"] = Coupling
        return Coupling
    if name == "Continuation":
        from tvbo.classes.continuation import Continuation

        globals()["Continuation"] = Continuation
        return Continuation
    if name in ("Function", "LossFunction"):
        from tvbo.classes.function import Function, LossFunction

        globals()["Function"] = Function
        globals()["LossFunction"] = LossFunction
        return globals()[name]
    if name == "Noise":
        from tvbo.classes.noise import Noise

        globals()["Noise"] = Noise
        return Noise
    if name == "SimulationExperiment":
        from tvbo.classes.experiment import SimulationExperiment

        globals()["SimulationExperiment"] = SimulationExperiment
        return SimulationExperiment
    if name == "SimulationStudy":
        from tvbo.classes.study import SimulationStudy

        globals()["SimulationStudy"] = SimulationStudy
        return SimulationStudy
    if name == "StudyCollection":
        from tvbo.classes.study import StudyCollection

        globals()["StudyCollection"] = StudyCollection
        return StudyCollection
    if name in ("Network", "Connectome"):
        from tvbo.classes.network import Network, Connectome

        globals()["Network"] = Network
        globals()["Connectome"] = Connectome
        return globals()[name]
    if name == "Atlas":
        from tvbo.classes.atlas import Atlas

        globals()["Atlas"] = Atlas
        return Atlas
    raise AttributeError(f"module 'tvbo.classes' has no attribute {name!r}")
