#
# Module: __init__.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
import glob
import os
from collections import namedtuple

from linkml_runtime.loaders import yaml_loader
from pybtex.database import parse_file

from tvbo import parse

ROOT = os.path.abspath(os.path.dirname(__file__))

bib_file = os.path.join(ROOT, "tvbo-literature-db.bib")

# Get all YAML files in the ROOT directory
yaml_files = glob.glob(os.path.join(ROOT, "*.yaml"))

# Dynamically assign module attributes for each YAML file
yaml_attributes = {}
for yaml_file in yaml_files:
    attribute_name = os.path.splitext(os.path.basename(yaml_file))[0]
    globals()[attribute_name] = yaml_file
    yaml_attributes[attribute_name] = yaml_file

# Create a namedtuple for accessing YAML files
YamlFiles = namedtuple("YamlFiles", yaml_attributes.keys())
study_metadata_files = YamlFiles(**yaml_attributes)


class SimulationStudies:
    def __init__(self):
        self.files = {}
        for path in yaml_files:
            yaml_data = yaml_loader.load_as_dict(path)
            key = yaml_data.pop("key")
            setattr(self, key, path)
            self.files[key] = path

    def load_all(self):
        for key, path in self.files.items():
            self.__setattr__(key, parse.metadata.load_simulation_study(path))

    def load(self, key):
        study = parse.metadata.load_simulation_study(self.files[key])
        self.__setattr__(key, study)
        return study


def load_study(citationkey: str):
    return parse.metadata.load_simulation_study(
        getattr(study_metadata_files, citationkey)
    )


def load_bibliography():
    return parse_file(bib_file)
