"""The runtime base relaxes two generated-model defaults, and nothing else.

`tvbo.classes.TVBOModel` is the base the runtime classes migrate onto (issue #69,
Part A). These tests pin the two deliberate differences from the generated
`ConfiguredBaseModel` so a regenerated datamodel cannot silently take them back,
and pin that the strictness guarding authored YAML stays on the datamodel classes.
"""

import pytest
from pydantic import ValidationError

from tvbo.classes._base import TVBOModel
from tvbo.datamodel.pydantic import ConfiguredBaseModel

RELAXED = {"extra": "allow", "validate_assignment": False}


class Runtime(TVBOModel):
    """A stand-in for a behaviour-carrying subclass."""

    name: str | None = None


def test_relaxes_exactly_the_two_documented_settings():
    generated = dict(ConfiguredBaseModel.model_config)
    runtime = dict(TVBOModel.model_config)
    differing = {k for k in generated.keys() | runtime.keys() if generated.get(k) != runtime.get(k)}
    assert differing == set(RELAXED)


def test_carries_every_other_generated_setting_unchanged():
    generated = dict(ConfiguredBaseModel.model_config)
    runtime = dict(TVBOModel.model_config)
    for key, value in generated.items():
        if key not in RELAXED:
            assert runtime[key] == value, f"{key} drifted from the generated config"


@pytest.mark.parametrize("setting,expected", sorted(RELAXED.items()))
def test_the_relaxations_have_the_documented_values(setting, expected):
    assert TVBOModel.model_config[setting] == expected


def test_a_runtime_cache_can_be_attached():
    """`extra="allow"`: codegen hangs caches off these objects."""
    obj = Runtime(name="x")
    obj._cache = {"parsed": 1}
    assert obj._cache == {"parsed": 1}


def test_assignment_is_not_revalidated():
    """`validate_assignment=False`: mutation in a hot path stays cheap."""
    obj = Runtime(name="x")
    obj.name = 123
    assert obj.name == 123


def test_the_datamodel_still_refuses_an_unknown_slot():
    """The guarantee that matters is at the specification boundary, and stays."""
    from tvbo.datamodel.pydantic import Function

    with pytest.raises(ValidationError):
        Function(name="f", not_a_slot="x")


def test_as_dict_drops_unset_slots():
    assert Runtime(name="x").as_dict() == {"name": "x"}
    assert "name" not in Runtime().as_dict()


def test_as_dict_excludes_runtime_state():
    """A cache is not a slot: ``extra='allow'`` must not leak one into a dump.

    The two settings interact — the relaxation that lets codegen hang a cache off one of
    these objects is the same one that would carry it into anything serialized from them,
    silently editing the specification that was authored.
    """
    obj = Runtime(name="x")
    obj.a_runtime_cache = object()
    assert obj.as_dict() == {"name": "x"}
    assert obj.a_runtime_cache is not None


def test_the_relaxations_survive_being_mixed_into_a_generated_class():
    """Guards the base ORDER, which is silent when wrong.

    Pydantic merges ``model_config`` across bases left to right and every generated class
    declares its own, so ``class X(TVBOModel, generated.X)`` — the intuitive mixin-first
    spelling — yields a class as strict as the datamodel, with no error anywhere. Only the
    generated class FIRST gives the runtime class its relaxations.
    """
    from tvbo.datamodel.pydantic import Function

    class RightWayRound(Function, TVBOModel):
        pass

    class WrongWayRound(TVBOModel, Function):
        pass

    for setting, expected in RELAXED.items():
        assert RightWayRound.model_config.get(setting) == expected, setting
    assert WrongWayRound.model_config.get("extra") == "forbid"
