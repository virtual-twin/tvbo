"""Factory constructors for :class:`Continuation`.

Loading a continuation from YAML, a string, or the curated database. Attached to the
generated classes by name (``ContinuationBehaviour`` -> ``Continuation``), so the
factories are available wherever the class is, including on a continuation nested inside
a loaded experiment.
"""

from __future__ import annotations

import os


class ContinuationBehaviour:
    """Load a continuation specification from YAML or the database."""

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "ContinuationBehaviour":
        """Load a Continuation from a YAML file.

        The file can be either a standalone Continuation YAML (root keys are Continuation
        fields), or a SimulationExperiment YAML containing a ``continuations`` section —
        in which case the *first* continuation entry is returned.
        """
        from tvbo.utils import yaml_loader

        return yaml_loader.load(str(path), cls)

    @classmethod
    def from_string(cls, yaml_string: str) -> "ContinuationBehaviour":
        """Create a Continuation from a YAML string."""
        from tvbo.utils import yaml_loader

        return yaml_loader.loads(yaml_string, target_class=cls)

    @classmethod
    def from_db(cls, name: str) -> "ContinuationBehaviour":
        """Load a Continuation by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Continuation", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available continuations in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("Continuation")
