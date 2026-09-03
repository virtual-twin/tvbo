#
# Module: software.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""The public import location for ``SimulationTool``.

The class is the generated one; what it does lives in :mod:`tvbo.behaviour.software`, attached to both generated forms, so a tool entry answers the same questions whether it came from LinkML's loader or from Pydantic validation.

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

from tvbo.datamodel import schema as tvbo_datamodel

SimulationTool = tvbo_datamodel.SimulationTool
