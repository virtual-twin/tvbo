#
# Module: misc_loader.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
This module deals with loading simulation components from different sources, other than the ontology and LEMS files.
"""
import yaml

from tvbo.knowledge import config
from tvbo.datamodel.tvbo_datamodel import SimulationStudy


def load_simulation_from_yaml(filename, experiment_key=None):
    sim_metadata = SimulationStudy(**yaml.safe_load(open(filename)))

    sim = config.config_sim_from_datamodel(sim_metadata, experiment_key=experiment_key)

    return sim
