#
# Module: __init__.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Access bundled simulation-study metadata and the literature bibliography.

Every `*.yaml` file in this package is discovered at import time and exposed both
as a module-level attribute and through the `study_metadata_files` namedtuple, each
mapping a study key to its file path. Helpers are provided to load individual studies
into [`SimulationStudy`](../classes/study.qmd) objects and to parse the accompanying
BibTeX database.
"""
import glob
import os
from collections import namedtuple
from functools import lru_cache

from linkml_runtime.loaders import yaml_loader

try:
    from pybtex.database import parse_file
except ImportError:
    parse_file = None  # pybtex is optional (docs extra)

ROOT = os.path.abspath(os.path.dirname(__file__))

# The package literature bibliography — single source of truth, shared with the
# ontology generators (scripts/ontology/_bib.py). Lives with the study database
# it annotates (tvbo/database/), keyed by citekey.
bib_file = os.path.abspath(os.path.join(ROOT, "..", "..", "database", "references.bib"))

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
    """Registry of the bundled simulation-study metadata files.

    On construction, every discovered YAML file is read to obtain its `key`, and that
    key is set as an attribute holding the file path. Calling [`load`](#load) or
    [`load_all`](#load_all) replaces a path with the loaded
    [`SimulationStudy`](../classes/study.qmd) object.

    Attributes:
        files: Mapping of study key to its YAML file path.
    """

    def __init__(self):
        self.files = {}
        for path in yaml_files:
            yaml_data = yaml_loader.load_as_dict(path)
            key = yaml_data.pop("key")
            setattr(self, key, path)
            self.files[key] = path

    def load_all(self):
        """Load every registered study, replacing each key's path with its object.

        After this call, each study key attribute holds a
        [`SimulationStudy`](../classes/study.qmd) instead of its file path.
        """
        from tvbo.classes.study import SimulationStudy

        for key, path in self.files.items():
            self.__setattr__(key, SimulationStudy.from_file(path))

    def load(self, key):
        """Load a single study by key and cache it on the registry.

        Args:
            key: Study key identifying the YAML file to load.

        Returns:
            The loaded [`SimulationStudy`](../classes/study.qmd), which is also
            stored as the `key` attribute on this registry.
        """
        from tvbo.classes.study import SimulationStudy

        study = SimulationStudy.from_file(self.files[key])
        self.__setattr__(key, study)
        return study


def load_study(citationkey: str):
    """Load a bundled study by its citation key.

    Args:
        citationkey: Name of the study's YAML file (without extension), matching an
            attribute of `study_metadata_files`.

    Returns:
        The [`SimulationStudy`](../classes/study.qmd) parsed from the matching file.
    """
    from tvbo.classes.study import SimulationStudy

    return SimulationStudy.from_file(getattr(study_metadata_files, citationkey))


@lru_cache(maxsize=1)
def load_bibliography():
    """Parse the bundled BibTeX literature database.

    Cached: the file ships with the package and cannot change within a process, while
    `get_citation` is called once per reference from inside template comprehensions — an
    eight-reference report re-read the same 45 KiB, 115-entry file eight times at ~30 ms each.

    Returns:
        The parsed `pybtex` bibliography for `tvbo/database/references.bib`.

    Raises:
        ImportError: If the optional `pybtex` dependency is not installed.
    """
    if parse_file is None:
        raise ImportError("pybtex is required for bibliography support. Install it with: pip install tvbo[docs]")
    return parse_file(bib_file)
