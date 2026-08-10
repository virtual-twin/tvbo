#
# Module: continuation.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""
Continuation
============
The public import location for :class:`Continuation`.

There is no wrapper class: the factory methods live in
:mod:`tvbo.behaviour.continuation` and are attached to the generated class itself, so a
continuation carries them however it was built — loaded from YAML, nested inside an
experiment, or constructed directly.

``Continuation()`` now requires a ``name``, as the schema always did. The wrapper used to
supply ``"continuation"`` as a default, which only ever applied to a bare call: the one
construction site in the package passes a name explicitly.

Usage
-----
>>> from tvbo import Continuation
>>> cont = Continuation.from_db("Generic2dOscillator-bifurcation")
>>> print(cont.name)
eq_in_I

>>> # Or add to an experiment
>>> exp = SimulationExperiment()
>>> exp.continuations[cont.name] = cont
"""

from tvbo.datamodel.schema import Continuation

__all__ = ["Continuation"]
