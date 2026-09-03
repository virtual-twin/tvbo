"""A backend names the edge attribute it needs at the declaration, instead of substituting zeros or an instantaneous graph.

`delayed` defaults to true, so a weights-only network under a stock coupling is the common case, and every backend here used to integrate it instantaneous and report success. `Network.carries` answers from the header and the explicit edges without reading a value; `edge_needs` states what a backend reads; `require_edge_attributes` refuses by name, saying what the network carries instead and how to fix it.
"""

import numpy as np
import pytest

from tvbo.adapters.base import edge_needs, require_edge_attributes
from tvbo.classes.network import Network
from tvbo.datamodel import tvbo_datamodel as model

W = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.2], [0.5, 0.2, 0.0]])
L = np.full((3, 3), 10.0)


def _edge(source, target, **params):
    return model.Edge(
        source=source, target=target, parameters={k: model.Parameter(name=k, value=v) for k, v in params.items()}
    )


def _experiment(tmp_path, network: str):
    from tvbo import SimulationExperiment

    spec = tmp_path / "exp.yaml"
    spec.write_text(
        f"""
id: 7
label: "edge requirements"
dynamics:
  name: Generic2dOscillator
  iri: tvbo:Generic2dOscillator
  parameters:
    a: {{value: -1.5}}
    b: {{value: -15.0}}
    c: {{value: 0.0}}
    d: {{value: 0.015}}
    e: {{value: 3.0}}
    f: {{value: 1.0}}
    tau: {{value: 4.0}}
    I: {{value: 0.1}}
  state_variables:
    V:
      equation: {{rhs: "d*tau*(I*gamma - V**3*f + V**2*e + V*g + V*local_coupling + W*alpha + c_glob*gamma)"}}
      coupling_variable: true
      initial_value: 0.0
    W:
      equation: {{rhs: "d*(V**2*c + V*b - W*beta + a)/tau"}}
      initial_value: 0.0
network:
{network}
integration:
  method: Heun
  step_size: 0.2
  duration: 10.0
"""
    )
    return SimulationExperiment.from_file(str(spec))


_DELAYED = """  number_of_nodes: 3
  coupling:
    c_lin:
      iri: tvbo:Linear
      parameters: {G: {value: 0.1}}
      incoming_states: [V]
"""
_UNDELAYED = _DELAYED.replace("      iri: tvbo:Linear\n", "      iri: tvbo:Linear\n      delayed: false\n")


class TestCarries:
    def test_from_the_matrices_in_hand(self):
        net = Network.from_matrix(W)
        assert net.carries("weight") and net.carries("weights")
        assert not net.carries("length") and not net.carries("tract_lengths") and not net.carries("gain")
        assert Network.from_matrix(W, L).carries("length")
        net.set_matrix("gain", np.ones((2, 3)))
        assert net.carries("gain")

    def test_from_the_explicit_edges(self):
        with_distance = Network(number_of_nodes=3, edges=[_edge(0, 1, weight=1.0, distance=2.0)])
        assert with_distance.carries("weight") and with_distance.carries("length")
        assert not with_distance.carries("delay"), "no edge declares a delay"
        zero_delay = Network(number_of_nodes=3, edges=[_edge(0, 1, delay=0.0)])
        assert not zero_delay.carries("delay") and not zero_delay.carries("length")
        assert Network(number_of_nodes=3, edges=[_edge(0, 1, delay=2.0)]).carries("delay")
        assert Network(number_of_nodes=3, edges=[_edge(0, 1, receptor=1)]).carries("receptor")

    def test_a_node_set_carries_weight_but_connects_nothing(self):
        nodes = Network(number_of_nodes=4)
        assert nodes.carries("weight"), "matrix('weight') answers with zeros on a node set"
        assert not nodes.has_connectome
        assert Network.from_matrix(W).has_connectome
        assert Network(number_of_nodes=3, edges=[_edge(0, 1)]).has_connectome
        assert not Network(number_of_nodes=3, edges=[model.Edge(label="weight")]).has_connectome, (
            "a template edge is not a connection"
        )


class TestEdgeNeeds:
    def test_weight_alone_when_nothing_is_delayed(self):
        assert edge_needs(Network.from_matrix(W), delayed=False) == [("the connectome", ("weight",))]
        assert edge_needs(Network.from_matrix(W), delayed=[]) == [("the connectome", ("weight",))]

    def test_a_delayed_coupling_needs_a_carrier_over_a_connectome(self):
        needs = edge_needs(Network.from_matrix(W), delay_carriers=("length", "delay"), delayed=["c_lin"])
        assert needs == [("the connectome", ("weight",)), ("the delayed coupling c_lin", ("length", "delay"))]
        assert edge_needs(Network(number_of_nodes=4), delayed=["c_lin"]) == [("the connectome", ("weight",))], (
            "a node set has nothing to delay"
        )

    def test_a_generated_graph_and_a_single_node_need_nothing(self):
        generated = Network(number_of_nodes=8, graph_generator=model.GraphGenerator(name="ring", type="WattsStrogatz"))
        assert edge_needs(generated, delayed=["c_lin"]) == []
        assert edge_needs(Network(number_of_nodes=1), delayed=["c_lin"]) == []
        assert edge_needs(None) == []

    def test_a_backend_requiring_more_says_which(self):
        needs = edge_needs(Network.from_matrix(W), required=("weight", "gain"), delayed=False)
        assert needs == [("the connectome", ("weight",)), ("the gain it lowers", ("gain",))]


class TestRequireEdgeAttributes:
    def test_refuses_by_name_and_says_what_is_carried(self):
        net = Network.from_matrix(W)
        needs = edge_needs(net, delay_carriers=("length", "delay"), delayed=["c_lin"])
        with pytest.raises(ValueError) as err:
            require_edge_attributes(net, "tvboptim", needs)
        message = str(err.value)
        for fragment in (
            "the tvboptim backend needs length or delay for the delayed coupling c_lin",
            "carries weight",
            "delayed: false",
            "set_matrix()",
        ):
            assert fragment in message, message

    def test_passes_when_any_carrier_is_present(self):
        needs = [("the delayed coupling c_lin", ("length", "delay"))]
        require_edge_attributes(Network.from_matrix(W, L), "tvboptim", needs)
        require_edge_attributes(Network(number_of_nodes=3, edges=[_edge(0, 1, delay=2.0)]), "tvboptim", needs)
        require_edge_attributes(object(), "tvboptim", needs)


class TestAdapters:
    def test_the_tvboptim_adapter_refuses_a_delayed_coupling_over_weights_alone(self, tmp_path):
        from tvbo.adapters.tvboptim import TvboptimAdapter

        exp = _experiment(tmp_path, _DELAYED)
        exp.network.set_matrix("weight", W)
        with pytest.raises(ValueError, match="the Tvboptim backend needs length or delay for the delayed coupling c_lin"):
            TvboptimAdapter(exp).refuse_unrenderable()
        exp.network.set_matrix("length", L)
        TvboptimAdapter(exp).refuse_unrenderable()

    def test_declaring_the_coupling_undelayed_is_the_other_fix(self, tmp_path):
        from tvbo.adapters.tvboptim import TvboptimAdapter

        exp = _experiment(tmp_path, _UNDELAYED)
        exp.network.set_matrix("weight", W)
        TvboptimAdapter(exp).refuse_unrenderable()

    def test_the_tvb_carrier_is_length_alone(self, tmp_path):
        from tvbo.adapters.base import BaseAdapter

        exp = _experiment(tmp_path, _DELAYED)
        exp.network.set_matrix("weight", W)
        exp.network.edges = list(exp.network.edges or []) + [_edge(0, 1, delay=2.0)]
        assert exp.network.carries("delay")
        with pytest.raises(ValueError, match="needs length for the delayed coupling c_lin"):
            BaseAdapter(exp).refuse_unrenderable()

    def test_the_standalone_tvboptim_graph_refuses_the_same_way(self):
        pytest.importorskip("tvboptim")
        from tvbo.adapters.tvboptim import to_tvboptim

        net = Network.from_matrix(W, coupling={"Linear": model.Coupling(iri="tvbo:Linear")})
        with pytest.raises(ValueError, match="length or delay"):
            to_tvboptim(net, return_type="graph")
        assert to_tvboptim(net, delays=False, return_type="graph") is not None
