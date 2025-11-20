#
# Module: constants.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
Constants
=========

This module contains the constants used in the project.
| Variable | Description |
|----------|----------|
| ROOT_DIR | Package directory |
| DATA_DIR | Data directory |

"""

from os.path import join, dirname, realpath, abspath

ROOT_DIR = abspath(join(dirname(realpath(__file__)), ".."))
DATA_DIR = join(ROOT_DIR, "data")
ONTO_DIR = join(DATA_DIR, "ontology", "tvb-o.owl")  # TODO: this is not used
SC_DIR = join(DATA_DIR, "normative_connectomes")


