#
# Module: software.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2
#
"""The public import location for ``SimulationTool``.

The class is the generated one; what it does lives in :mod:`tvbo.behaviour.software`, attached to the generated class, so a tool entry answers the same questions however it was built.

Usage
-----
>>> from tvbo.classes.software import SimulationTool
>>> lems = SimulationTool.for_format("neuroml")
>>> lems.name
'LEMS'
>>> lems.dimension_of("mV"), lems.symbol_of("Hz")
('voltage', 'per_s')
"""

from __future__ import annotations

from tvbo import datamodel as tvbo_datamodel

SimulationTool = tvbo_datamodel.SimulationTool
