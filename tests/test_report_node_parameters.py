"""A parameter the network sets region by region is reported by the span it takes, not by the model value no region holds."""

from types import SimpleNamespace

from tvbo.datamodel.schema import Dynamics, Node, Parameter
from tvbo.utils.report import node_parameters, symbol_table


def _experiment(omegas):
    nodes = [Node(id=i, label=f"r{i}", parameters={"omega": Parameter(name="omega", value=w)}) for i, w in enumerate(omegas)]
    return SimpleNamespace(network=SimpleNamespace(nodes=nodes))


def _model():
    return Dynamics(name="Osc", parameters={"omega": Parameter(name="omega", value=1.0), "a": Parameter(name="a", value=-0.5)})


def test_the_span_over_the_regions_is_collected():
    assert node_parameters([_experiment([0.04, 0.07, 0.05])]) == {"omega": "per region, 0.04 to 0.07"}


def test_the_symbol_table_prints_the_span_and_keeps_the_rest():
    table = symbol_table(_model(), per_node=node_parameters([_experiment([0.04, 0.07])]))
    assert "per region, 0.04 to 0.07" in table
    assert "parameter (per region)" in table
    assert "-0.5" in table


def test_a_sweep_still_wins():
    table = symbol_table(_model(), swept={"omega": "[0, 1], n=3"}, per_node=node_parameters([_experiment([0.04, 0.07])]))
    assert "[0, 1], n=3" in table and "per region" not in table
