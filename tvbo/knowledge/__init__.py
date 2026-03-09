#
# Module: __init__.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

from . import ontology, query
from .simulation.localdynamics import Dynamics
from .simulation.network import Coupling
from .simulation.integration import Integrator
from tvbo.data.tvbo_data.connectomes import Connectome

Model = Dynamics  # Backwards compatibility
LocalDynamics = Dynamics  # Backwards compatibility
