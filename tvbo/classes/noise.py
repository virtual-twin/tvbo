#
# Module: noise.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""The public import location for ``Noise`` and ``Integrator``.

Both are the generated classes. What they do lives in :mod:`tvbo.behaviour.noise` and :mod:`tvbo.behaviour.integrator`, attached where the classes are generated, so a record carries its behaviour however it was built — and, for `Noise`, is a JAX pytree from the moment the class exists rather than from whenever some module imported it.
"""

from __future__ import annotations

import functools

from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.ontology.owl import onto

Noise = tvbo_datamodel.Noise
Integrator = tvbo_datamodel.Integrator


@functools.cache
def _available_integrators():
    """The ontology's integration-method classes, resolved and memoised on first use."""
    return onto.IntegrationMethod.descendants(include_self=False)


def __getattr__(name):  # PEP 562: keep ``available_integrators`` importable, lazily.
    """Resolve ``available_integrators`` on first access, so importing this module does not load the ontology."""
    if name == "available_integrators":
        return _available_integrators()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
