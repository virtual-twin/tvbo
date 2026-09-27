"""Factory constructors for :class:`Continuation`.

Loading a continuation from YAML, a string, or the curated database — the constructors every catalogued class shares, from :class:`tvbo.behaviour._runtime.Catalogued`. Attached to the generated classes by name (``ContinuationBehaviour`` -> ``Continuation``), so the factories are available wherever the class is, including on a continuation nested inside a loaded experiment.
"""

from __future__ import annotations

from tvbo.behaviour._runtime import Catalogued


class ContinuationBehaviour(Catalogued):
    """Load a continuation specification from YAML or the database."""
