"""Uniform metadata resolution: 3 routes, YAML supervenes, field-level enrich.

Covers the ``iri``-sourced-with-inline-override contract:
- a spec referenced by ``iri`` is filled from the registry entry it points to;
- inline metadata supervenes at the *leaf* (override only ``parameters.a.value``,
  keep every sibling parameter and the equations from the source);
- this holds for the top-level ``dynamics`` slot, the keyed ``network.dynamics``
  entries, and a directly-constructed ``Dynamics`` / ``Coupling``.
"""

import pytest

from tvbo.utils import deep_merge


# deep_merge — the field-level precedence engine
def test_deep_merge_leaf_override_keeps_siblings():
    base = {"parameters": {"a": {"value": 0, "unit": "mV"}, "b": {"value": 2}}}
    override = {"parameters": {"a": {"value": 1}}}
    out = deep_merge(base, override)
    assert out["parameters"]["a"]["value"] == 1  # leaf overridden
    assert out["parameters"]["a"]["unit"] == "mV"  # sibling field kept
    assert out["parameters"]["b"]["value"] == 2  # sibling param kept


def test_deep_merge_does_not_mutate_inputs():
    base = {"x": {"y": 1}}
    override = {"x": {"z": 2}}
    deep_merge(base, override)
    assert base == {"x": {"y": 1}}
    assert override == {"x": {"z": 2}}


# Dynamics
def test_dynamics_iri_only_full_population():
    from tvbo import Dynamics

    d = Dynamics(iri="tvbo:Generic2dOscillator")
    assert d.name == "Generic2dOscillator"
    assert len(d.parameters) == 12
    assert len(d.state_variables) == 2


def test_dynamics_iri_with_partial_override():
    from tvbo import Dynamics

    d = Dynamics(iri="tvbo:Generic2dOscillator", parameters={"a": {"value": 7}})
    assert d.parameters["a"].value == 7  # inline wins
    assert d.parameters["b"].value == -10.0  # sibling from registry
    assert len(d.parameters) == 12  # all siblings kept
    assert len(d.state_variables) == 2  # equations from registry


def test_dynamics_from_db_backcompat():
    from tvbo import Dynamics

    d = Dynamics.from_db("JansenRit")
    assert d.name == "JansenRit"
    assert len(d.parameters) >= 10


# SimulationExperiment — top-level dynamics + keyed network.dynamics
def _exp(**net):
    from tvbo import SimulationExperiment

    net.setdefault("network", {}).setdefault("coupling", {"long_range": {"iri": "tvbo:Linear"}})
    return SimulationExperiment(
        integration={"method": "Heun", "noise": None},
        **net,
    )


def test_experiment_top_level_dynamics_override():
    exp = _exp(dynamics={"iri": "tvbo:Generic2dOscillator", "parameters": {"a": {"value": 1}}})
    assert exp.dynamics.name == "Generic2dOscillator"
    assert exp.dynamics.parameters["a"].value == 1
    assert exp.dynamics.parameters["b"].value == -10.0
    assert len(exp.dynamics.parameters) == 12


def test_experiment_network_dynamics_keyed_override():
    exp = _exp(
        network={
            "number_of_nodes": 2,
            "dynamics": {"g2d": {"iri": "tvbo:Generic2dOscillator", "parameters": {"a": {"value": 1}}}},
        }
    )
    g2d = exp.network.dynamics["g2d"]
    assert g2d.name == "g2d"  # keyed identifier preserved
    assert str(g2d.iri) == "tvbo:Generic2dOscillator"
    assert g2d.parameters["a"].value == 1  # inline wins
    assert g2d.parameters["b"].value == -10.0  # sibling from registry
    assert len(g2d.parameters) == 12


# Coupling
@pytest.mark.parametrize(
    "curie,expected",
    [
        ("tvbo:Linear", "Linear"),
        ("tvbo:Sigmoidal", "Sigmoidal"),
        ("tvbo:KuramotoCoupling", "KuramotoCoupling"),
    ],
)
def test_coupling_iri_resolves_by_local_name(curie, expected):
    """An ``iri``-only ``Coupling`` resolves to the registry entry named by the CURIE local name, never to the default coupling."""
    from tvbo.classes.coupling import Coupling

    c = Coupling(iri=curie)
    assert c.name == expected
    assert c.pre_expression is not None


def test_coupling_explicit_name_preserved():
    """A named coupling keeps its name and is filled from the function its ``iri`` names.

    Naming it makes the record a definition, so construction expands nothing — that stays the cheap, local step. ``enrich()`` is where it draws on the function it says it is an instance of, and a ``network.coupling`` entry gets that at experiment load.
    """
    from tvbo.classes.coupling import Coupling

    c = Coupling(name="MyCustom", iri="tvbo:Sigmoidal")
    assert c.name == "MyCustom"
    assert not c.parameters

    c.enrich()
    assert c.name == "MyCustom"  # explicit name still wins
    assert len(c.parameters) == 5  # params now from Sigmoidal


@pytest.mark.parametrize("curie", ["tvbo:Linear", "tvbo:Sigmoidal", "tvbo:SigmoidalJansenRit"])
def test_both_generated_forms_resolve_an_iri_the_same_way(curie):
    """One YAML cannot mean two different couplings depending on which loader read it.

    The behaviour is attached to both generated forms, but they do not share a construction hook — the dataclasses call ``__post_init__``, the Pydantic models ``model_post_init``. With only the first, everything loaded through ``tvbo.utils.pydantic_loader`` came back as a bare default ``Linear``, unpopulated, with no error to say so.
    """
    from tvbo.datamodel import pydantic as pdm
    from tvbo.datamodel import schema

    dataclass_form = schema.Coupling(iri=curie)
    pydantic_form = pdm.Coupling(iri=curie)

    assert dataclass_form.name == pydantic_form.name
    assert str(dataclass_form.pre_expression.rhs) == str(pydantic_form.pre_expression.rhs)
    assert str(dataclass_form.post_expression.rhs) == str(pydantic_form.post_expression.rhs)
    assert list(dataclass_form.parameters) == list(pydantic_form.parameters)
    assert [p.value for p in dataclass_form.parameters.values()] == [p.value for p in pydantic_form.parameters.values()]
