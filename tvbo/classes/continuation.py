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
Runtime wrapper around the auto-generated ``Continuation`` dataclass.

Provides ``from_file`` / ``from_string`` factory methods so that a continuation specification can be loaded from YAML just like ``Dynamics``.

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

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tvbo.utils import yaml_loader

from tvbo.datamodel import schema as tvbo_datamodel

if TYPE_CHECKING:
    pass


class Continuation(tvbo_datamodel.Continuation):
    """Runtime continuation specification with factory constructors.

    Inherits all schema-defined fields (``free_parameters``, ``ds``,
    ``max_steps``, ``branches``, etc.) from the generated datamodel and adds convenience factory methods.
    """

    def __init__(self, name="continuation", **kwargs):
        if name is not None:
            kwargs["name"] = str(name)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_datamodel(cls, cont_meta: tvbo_datamodel.Continuation) -> "Continuation":
        """Create from an auto-generated datamodel instance."""
        return cls(**cont_meta._as_dict)

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "Continuation":
        """Load a Continuation from a YAML file.

        The file can be either:
        1. A standalone Continuation YAML (root keys are Continuation fields).
        2. A SimulationExperiment YAML that contains a ``continuations`` section — in which case the *first* continuation entry is returned.

        Parameters
        ----------
        path : str or PathLike
            Path to the YAML file.

        Returns
        -------
        Continuation
            The loaded continuation specification.
        """
        return yaml_loader.load(str(path), cls)

    @classmethod
    def from_string(cls, yaml_string: str) -> "Continuation":
        """Create a Continuation from a YAML string.

        Parameters
        ----------
        yaml_string : str
            YAML-formatted string defining the continuation.

        Returns
        -------
        Continuation
        """
        return yaml_loader.loads(yaml_string, target_class=cls)

    @classmethod
    def from_db(cls, name: str) -> "Continuation":
        """Load a Continuation by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Continuation", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available continuations in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("Continuation")
