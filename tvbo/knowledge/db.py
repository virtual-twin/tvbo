#
# Module: db.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
import glob
import os

from linkml_runtime.loaders import yaml_loader
from tvbo.utils import Bunch
from tvbo.knowledge import constants
from tvbo.datamodel.tvbo_datamodel import SimulationStudy

DATA_DIR = os.path.join(constants.DATA_DIR, "db")

EXAMPLE_FILES = glob.glob(os.path.join(DATA_DIR, "*.yaml"))

# print(EXAMPLE_FILES)
# Load data from YAML files and insert into the database
SimulationStudies = Bunch()
for path in EXAMPLE_FILES:
    yaml_data = yaml_loader.load_as_dict(path)
    key = yaml_data.pop("key")
    SimulationStudies[key] = yaml_loader.load(path, target_class=SimulationStudy)
    # simulation_study = st(**yaml_data)
