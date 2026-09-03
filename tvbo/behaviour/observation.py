#
# Module: behaviour/observation.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""How an ``Observation`` record is loaded — from a file, or by name from the tvbo database."""

from __future__ import annotations


class ObservationBehaviour:
    """A declared observation model's factory constructors, on both generated forms."""

    @classmethod
    def from_file(cls, path: str):
        """Load an Observation from a YAML file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(str(path), target_class=cls)

    @classmethod
    def from_db(cls, name: str):
        """Load an Observation by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Observation", name)))
