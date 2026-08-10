"""The contract the behaviour-mixin design rests on (issue #69, Part A).

Behaviour is attached to the generated Pydantic classes as a plain mixin base rather
than through a runtime subclass, and the state that behaviour needs — caches, resolved
handles — rides on private attributes. That works only if four things hold of the
generated models, none of which this project controls: they belong to Pydantic and to
LinkML's generator.

They are pinned here because each one, if it silently stopped holding, would not fail
loudly. A cache that starts serializing corrupts the specification it is written back
into; a mixin that stops resolving takes the whole public API (``d.symbolic``,
``d.plot()``) with it.
"""

import pytest
from pydantic import ValidationError

from tvbo.datamodel.pydantic import Dynamics


def test_a_cache_rides_on_a_private_attribute():
    """Behaviour state attaches without declaring a slot, under the strict default.

    This is what makes ``extra="allow"`` unnecessary: relaxing it was justified as the
    way to hang runtime state off a spec, and the strict config already allows exactly
    that, without opening the spec to stray fields.
    """
    spec = Dynamics(name="x")
    spec._behaviour_cache = {"heavy": "object"}
    assert spec._behaviour_cache == {"heavy": "object"}


def test_a_cache_never_reaches_a_dump():
    """A cache is not a slot, and must not survive into anything serialized."""
    spec = Dynamics(name="x")
    spec._behaviour_cache = {"heavy": "object"}
    assert "_behaviour_cache" not in spec.model_dump()
    assert not any(key.startswith("_") for key in spec.model_dump())


def test_a_spec_still_refuses_an_undeclared_slot():
    """The guarantee that matters is at the specification boundary, and stays.

    A typo'd slot in an authored recipe has to fail loudly; that is the whole reason the
    generated config forbids extras.
    """
    assert Dynamics.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError):
        Dynamics(name="x", not_a_slot=1)
    with pytest.raises(ValueError):
        Dynamics(name="x").not_a_slot = 1


def test_a_plain_mixin_supplies_behaviour_to_a_generated_model():
    """A non-model base contributes methods and properties, and costs the spec nothing.

    Plain rather than a ``BaseModel`` subclass on purpose: two models in the bases would
    merge their ``model_config`` left to right, so a mixin carrying one could silently
    re-open ``extra`` or change validation for every class it is attached to.
    """

    class Behaviour:
        @property
        def described(self):
            return f"<{self.name}>"

    class WithBehaviour(Behaviour, Dynamics):
        pass

    spec = WithBehaviour(name="x")
    assert spec.described == "<x>"
    assert spec.model_config["extra"] == "forbid"
    assert spec.model_dump() == Dynamics(name="x").model_dump()
