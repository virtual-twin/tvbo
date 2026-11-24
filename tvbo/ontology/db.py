# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Load bundled example `SimulationStudy` definitions from the ontology data directory.

At import time this module scans the `db` data directory for YAML files, loads each into a [`SimulationStudy`](../datamodel/pydantic.qmd) instance, and exposes them keyed by their `key` field through the module-level `SimulationStudies` mapping.
"""

import glob
import os

from tvbo.datamodel import SimulationStudy
from tvbo.ontology import constants
from tvbo.utils import Bunch, yaml_loader

DATA_DIR = os.path.join(constants.DATA_DIR, "db")

EXAMPLE_FILES = glob.glob(os.path.join(DATA_DIR, "*.yaml"))

# Load data from YAML files and insert into the database
SimulationStudies = Bunch()
for path in EXAMPLE_FILES:
    yaml_data = yaml_loader.load_as_dict(path)
    key = yaml_data.pop("key")
    SimulationStudies[key] = yaml_loader.load(path, target_class=SimulationStudy)
