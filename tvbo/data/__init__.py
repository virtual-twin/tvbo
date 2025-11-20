#
# Module: __init__.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""
Data Module
===========

Access and manage TVB-O data.
"""
from .db import *
from .tvbo_data import *

from .tvbo_data.connectomes import Connectome
from .tvbo_data.atlases import Atlas
