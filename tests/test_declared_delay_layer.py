"""A network's delays are an edge layer of their own, and a run integrates the layer the network declares.

Unless declared, a connection's delay is its tract length over the conduction speed. A network may declare it instead: as a stored ``delay`` layer in a time unit, or by a ``transforms:`` entry on ``delay`` whose equation rewrites the derived delay, where ``dt`` names the integration step. A declared delay is what every reader gets: the run, the longest delay the history is sized by, a frozen copy. A backend that derives its delays from tract lengths itself refuses a network that declares them, and a network that declares none renders the code it rendered before.

Each run is compared with the same network whose tract lengths were set so that length over speed gives the declared delays, or with a forward Euler written out here in which every edge reads its source a whole number of steps before the current state.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tvboptim")
pytestmark = pytest.mark.backend_tvboptim

from tvbo import SimulationExperiment
from tvbo.adapters.base import BaseAdapter

N, DT, SPEED, N_STEPS = 5, 1.0, 2.0, 40
TAU, DRIVE, GAIN, START = 8.0, 0.3, 1.5, 0.1
RENDERS = Path(__file__).parent / "reference_data" / "declared_delay_layer"

SPEC = {
    "id": 1,
    "label": "declared delay layer",
    "dynamics": {
        "name": "Leaky",
        "parameters": {"tau": {"value": TAU}, "I": {"value": DRIVE}},
        "coupling_inputs": {"c": {"name": "c"}},
        "state_variables": {
            "x": {
                "equation": {"lhs": "Derivative(x, t)", "rhs": "(-x + I + c) / tau"},
                "coupling_variable": True,
                "variable_of_interest": True,
                "initial_value": START,
            }
        },
        "output": ["x"],
        "number_of_modes": 1,
    },
    "network": {
        "number_of_nodes": N,
        "nodes": [{"id": i, "label": f"r{i}"} for i in range(N)],
        "parameters": {"conduction_speed": {"name": "conduction_speed", "value": SPEED, "unit": "mm_per_ms"}},
        "coupling": {
            "lin": {
                "label": "lin",
                "delayed": True,
                "parameters": {"G": {"name": "G", "value": GAIN}},
                "pre_expression": {"rhs": "x_j"},
                "post_expression": {"rhs": "G * gx"},
                "incoming_states": ["x"],
            }
        },
    },
    "integration": {"method": "euler", "duration": N_STEPS * DT, "step_size": DT, "transient_time": 0.0},
}
"""Five leaky nodes that start equal and are driven apart by unequal incoming weights, under a linear coupling declared delayed."""

HALF_UP = "(Max(floor(delay / dt + 0.5), 1) - 1) * dt"
"""The delay of a ring buffer of ``max(round(delay / dt), 1)`` steps that is read before the step writes to it: one step shorter, and none at all where the buffer holds a single step."""


def _connectome(every_pair=False):
    """Weights with unequal row sums and each connection's delay as a whole number of steps from one to six; with *every_pair* no off-diagonal weight is zero."""
    i, j = np.indices((N, N))
    weights = 0.05 + 0.03 * ((3 * i + 5 * j) % 7)
    if not every_pair:
        weights[(i + 2 * j) % 4 == 0] = 0.0
    weights[i == j] = 0.0
    steps = 1 + (2 * i + 3 * j) % 6
    steps[i == j] = 0
    return weights, steps


PATHS = {"some pairs unconnected": False, "every pair connected": True}
"""Under three quarters of the entries connected, the rendered coupling visits the stored edges; above, it takes tvboptim's own path over all pairs."""


def _declared(form, steps):
    """What the network declares under *form*, and the whole steps each connection is then delayed by."""
    if form == "a stored delay layer":
        other = np.where(steps > 0, 8 - steps, 0)
        return {"delay": other * DT}, other
    if form == "a transform that doubles the delay":
        return {"transforms": [{"name": "delay", "equation": {"rhs": "delay * 2"}}]}, 2 * steps
    return {"transforms": [{"name": "delay", "equation": {"rhs": HALF_UP}}]}, np.maximum(steps, 1) - 1


FORMS = ("a stored delay layer", "a transform that doubles the delay", "a transform that reads the integration step")


def _experiment(weights, lengths=None, delay=None, transforms=None, edge_delay=None, explorations=None, spec=None, **network):
    """The experiment *spec* (`SPEC` unless given) over *weights* with tract lengths, a ``delay`` layer, transforms, per-edge delays, explorations and further network slots as given."""
    spec = copy.deepcopy(spec or SPEC)
    spec["network"].update(network)
    if explorations:
        spec["explorations"] = explorations
    if transforms:
        spec["network"]["transforms"] = transforms
    if edge_delay is not None:
        spec["network"]["edges"] = [
            {
                "source": int(j),
                "target": int(i),
                "directed": True,
                "parameters": {"weight": {"value": float(weights[i, j])}, "delay": {"value": float(edge_delay[i, j])}},
            }
            for i, j in np.argwhere(weights != 0)
        ]
        return SimulationExperiment(**spec)
    exp = SimulationExperiment(**spec)
    exp.network.set_matrix("weight", weights)
    if lengths is not None:
        exp.network.set_matrix("length", lengths)
    if delay is not None:
        exp.network.set_matrix("delay", delay)
    return exp


def _run(exp):
    """The integrated state as ``(time, node)``, its first row the state after one step."""
    return np.asarray(exp.run(format="tvboptim").integration.data).reshape(-1, N)


def _by_length(weights, steps):
    """The run of the network whose tract lengths give *steps* at its conduction speed."""
    return _run(_experiment(weights, steps * SPEED * DT))


def _euler(weights, steps):
    """Forward Euler with every edge reading its source *steps* steps before the current state, on a history held at the start value; rows as `_run` returns them."""
    source = np.arange(N)[None, :].repeat(N, axis=0)
    states = [np.full(N, START)]
    for n in range(N_STEPS):
        past = np.stack(states)[np.clip(n - steps, 0, None), source]
        states.append(states[-1] + DT * (-states[-1] + DRIVE + GAIN * (weights * past).sum(axis=1)) / TAU)
    return np.stack(states[1:])


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("form", FORMS)
def test_a_run_integrates_the_delays_the_network_declares(form, path):
    weights, steps = _connectome(PATHS[path])
    declaration, declared_steps = _declared(form, steps)
    run = _run(_experiment(weights, steps * SPEED * DT, **declaration))
    undeclared = _by_length(weights, steps)
    assert np.abs(_by_length(weights, declared_steps) - undeclared).max() > 1e-3, "the declared delays move the run"
    assert np.abs(run - undeclared).max() > 1e-3, f"with {form} the run is the one of tract length over speed"
    np.testing.assert_allclose(run, _by_length(weights, declared_steps), rtol=0, atol=1e-12)


@pytest.mark.parametrize("beside", ["no tract lengths", "tract lengths that give other delays"])
@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_a_stored_delay_layer_delays_the_coupling_on_either_graph(representation, beside):
    weights, steps = _connectome()
    other = _declared("a stored delay layer", steps)[1]
    lengths = None if beside == "no tract lengths" else steps * SPEED * DT
    exp = _experiment(weights, lengths, delay=other * DT, graph_representation=representation)
    np.testing.assert_allclose(_run(exp), _euler(weights, other), rtol=0, atol=1e-12)


def test_a_delay_layer_stored_sparse_is_integrated_as_the_dense_one():
    from scipy import sparse

    weights, steps = _connectome()
    other = _declared("a stored delay layer", steps)[1]
    exp = _experiment(weights, steps * SPEED * DT)
    exp.network.set_matrix("delay", sparse.csr_matrix(other * DT))
    assert exp.max_delay == pytest.approx(other.max() * DT)
    np.testing.assert_allclose(_run(exp), _euler(weights, other), rtol=0, atol=1e-12)


CARRIERS = ("tract lengths", "per-edge delays", "a stored delay layer", "a transform that reads the integration step")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("carrier", CARRIERS)
def test_an_edge_delayed_by_no_step_reads_the_current_state(carrier, path):
    weights, steps = _connectome(PATHS[path])
    shorter = np.maximum(steps, 1) - 1
    assert ((shorter == 0) & (weights != 0)).any() and shorter.max() > 0, (
        "some connected edges have no delay and some have one"
    )
    if carrier == "tract lengths":
        exp = _experiment(weights, shorter * SPEED * DT)
    elif carrier == "per-edge delays":
        exp = _experiment(weights, edge_delay=shorter * DT)
    elif carrier == "a stored delay layer":
        exp = _experiment(weights, steps * SPEED * DT, delay=shorter * DT)
    else:
        exp = _experiment(weights, steps * SPEED * DT, **_declared(carrier, steps)[0])
    np.testing.assert_allclose(_run(exp), _euler(weights, shorter), rtol=0, atol=1e-12)


@pytest.mark.parametrize("form", FORMS)
def test_the_longest_delay_is_the_longest_declared_one(form):
    weights, steps = _connectome()
    declaration, declared_steps = _declared(form, steps)
    exp = _experiment(weights, steps * SPEED * DT, **declaration)
    assert exp.max_delay == pytest.approx(declared_steps.max() * DT)


def test_a_delay_layer_is_read_in_the_time_unit_it_declares():
    weights, steps = _connectome()
    other = _declared("a stored delay layer", steps)[1]
    exp = _experiment(weights, steps * SPEED * DT, delay=other * DT / 1000.0, edges=[{"label": "delay", "unit": "s"}])
    np.testing.assert_allclose(_run(exp), _by_length(weights, other), rtol=0, atol=1e-12)


def test_a_delay_layer_in_a_unit_that_is_no_time_is_refused():
    weights, steps = _connectome()
    exp = _experiment(weights, steps * SPEED * DT, delay=steps * DT, edges=[{"label": "delay", "unit": "mm"}])
    with pytest.raises(ValueError, match="delay"):
        exp.render("tvboptim")


@pytest.mark.parametrize("form", ("no declaration", *FORMS))
def test_a_frozen_copy_integrates_the_delays_of_its_source(form, tmp_path):
    weights, steps = _connectome()
    declaration, declared_steps = ({}, steps) if form == "no declaration" else _declared(form, steps)
    exp = _experiment(weights, steps * SPEED * DT, **declaration)
    (tmp_path / "experiment.yaml").write_text(exp.freeze_yaml(tmp_path))
    frozen = SimulationExperiment.from_file(str(tmp_path / "experiment.yaml"))
    np.testing.assert_allclose(_run(frozen), _by_length(weights, declared_steps), rtol=0, atol=1e-12)


def _refusals(exp):
    """The calls by which a backend that derives its delays from tract lengths meets a declaration, by name."""
    return {
        "the default adapter": lambda: BaseAdapter(exp).refuse_unrenderable(),
        "the jax render": lambda: exp.render("jax"),
        "the tvb render": lambda: exp.render("tvb"),
    }


@pytest.mark.parametrize("backend", ["the default adapter", "the jax render", "the tvb render"])
@pytest.mark.parametrize("form", FORMS[:2])
def test_a_backend_that_reads_tract_lengths_refuses_a_declared_delay(form, backend):
    weights, steps = _connectome()
    _refusals(_experiment(weights, steps * SPEED * DT))[backend]()
    declaration, _ = _declared(form, steps)
    with pytest.raises(NotImplementedError, match="declares its delays itself"):
        _refusals(_experiment(weights, steps * SPEED * DT, **declaration))[backend]()


SPEED_AXIS = {
    "sweep": {
        "name": "sweep",
        "mode": "product",
        "space": [{"parameter": "network.conduction_speed", "explored_values": [1.0, 2.0]}],
    }
}
"""A sweep of the conduction speed, which moves the delays of a network that derives them from its tract lengths and nothing in one that stores them."""


def test_a_conduction_speed_axis_over_a_stored_delay_layer_is_refused():
    weights, steps = _connectome()
    _experiment(weights, steps * SPEED * DT, explorations=SPEED_AXIS).run(format="tvboptim")
    with pytest.raises((ValueError, NotImplementedError), match="conduction_speed"):
        _experiment(weights, steps * SPEED * DT, delay=steps * DT, explorations=SPEED_AXIS).run(format="tvboptim")


def _undeclared(case):
    """A network that declares no delay of its own, by how its delays arise."""
    weights, steps = _connectome()
    if case == "tract_lengths":
        return _experiment(weights, steps * SPEED * DT)
    if case == "per_edge_delays":
        return _experiment(weights, edge_delay=steps * DT)
    exp = _experiment(weights)
    for coupling in exp.network.coupling.values():
        coupling.delayed = False
    return exp


@pytest.mark.parametrize("case", ["tract_lengths", "per_edge_delays", "undelayed"])
def test_a_network_that_declares_no_delay_renders_the_code_it_rendered(case):
    assert _undeclared(case).render("tvboptim") == (RENDERS / f"{case}.py.txt").read_text()


def test_a_delay_layer_on_a_network_without_tract_lengths_is_not_the_undelayed_run():
    weights, steps = _connectome()
    run = _run(_experiment(weights, delay=steps * DT))
    assert np.abs(run - _euler(weights, np.zeros_like(steps))).max() > 1e-3, "the run is the undelayed network's"
    np.testing.assert_allclose(run, _euler(weights, steps), rtol=0, atol=1e-12)


def _declaring(form, listed_edges=False):
    """The network of `_connectome` with a declaration of *form* on it, its connections stored as matrices with tract lengths or, with *listed_edges*, listed one by one with a delay each."""
    weights, steps = _connectome()
    declaration, _ = _declared(form, steps)
    if not listed_edges:
        return _experiment(weights, steps * SPEED * DT, **declaration)
    exp = _experiment(weights, edge_delay=steps * DT, transforms=declaration.get("transforms"))
    if "delay" in declaration:
        exp.network.set_matrix("delay", declaration["delay"])
    return exp


def _brian2(exp):
    from tvbo.adapters.brian2 import Brian2Adapter

    return Brian2Adapter(exp)


READERS = {
    "the tvb run": (False, "tvb", lambda exp: exp.run(format="tvb")),
    "the jax run": (False, None, lambda exp: exp.run(format="jax")),
    "the rateml-cuda render": (False, None, lambda exp: exp.render("rateml-cuda")),
    "the rateml-driver render": (False, None, lambda exp: exp.render("rateml-driver")),
    "the ModelingToolkit render": (False, None, lambda exp: exp.render("modelingtoolkit")),
    "the Brian2 render": (False, None, lambda exp: exp.render("brian2")),
    "the Brian2 run": (False, "brian2", lambda exp: _brian2(exp).run()),
    "the NeuroML render": (True, None, lambda exp: exp.render("neuroml")),
    "the LEMS render": (True, None, lambda exp: exp.render("lems")),
    "the PyRates YAML": (False, None, lambda exp: exp.render("pyrates-yaml")),
    "the PyRates network YAML": (False, None, lambda exp: exp.network.to_yaml(format="pyrates")),
    "the PyRates run of listed edges": (True, "pyrates", lambda exp: exp.run(format="pyrates")),
}
"""Every further reader that takes a delay from tract lengths or per-edge values in its own code: whether it is asked over listed edges, the only delays NeuroML and a small PyRates circuit read, the package its call imports before it reads the network, and the call."""


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("form", FORMS[:2])
def test_a_reader_of_its_own_delays_names_the_declaration_it_would_drop(form, reader):
    listed_edges, package, read = READERS[reader]
    if package:
        pytest.importorskip(package)
    with pytest.raises(NotImplementedError, match="declares its delays itself") as refusal:
        read(_declaring(form, listed_edges))
    assert "or on pyrates where the network lists no edge between two nodes" in str(refusal.value)


WITHOUT_LENGTHS = {
    "the default adapter": (None, lambda exp: BaseAdapter(exp).refuse_unrenderable()),
    "the jax render": (None, lambda exp: exp.render("jax")),
    "the tvb render": (None, lambda exp: exp.render("tvb")),
    **{name: (package, read) for name, (listed_edges, package, read) in READERS.items() if not listed_edges},
}
"""Every reader of tract lengths, asked over a network that stores its delays and measures no tract length."""


@pytest.mark.parametrize("reader", WITHOUT_LENGTHS)
def test_a_reader_names_the_declaration_before_the_tract_lengths_it_would_ask_for(reader):
    package, read = WITHOUT_LENGTHS[reader]
    if package:
        pytest.importorskip(package)
    weights, steps = _connectome()
    with pytest.raises(NotImplementedError, match="declares its delays itself"):
        read(_experiment(weights, delay=steps * DT))


LEAVES = {
    "an exploration axis on the conduction speed": {"explorations": SPEED_AXIS},
    "an exploration axis on the tract lengths": {
        "explorations": {
            "sweep": {
                "name": "sweep",
                "mode": "product",
                "space": [{"parameter": "network.edges.length", "explored_values": [1.0, 2.0]}],
            }
        }
    },
    "a fitted conduction speed": {
        "optimizations": {"fit": {"name": "fit", "free_parameters": [{"parameter": "network.conduction_speed"}]}}
    },
}
"""What moves the delays of a network that derives them from tract lengths while a run is under way, and nothing in one that declares them."""


@pytest.mark.parametrize("leaf", LEAVES)
@pytest.mark.parametrize("form", FORMS)
def test_a_declared_delay_leaves_no_speed_or_length_to_sweep_or_fit(form, leaf):
    weights, steps = _connectome()
    exp = _experiment(weights, steps * SPEED * DT, spec={**SPEC, **LEAVES[leaf]}, **_declared(form, steps)[0])
    with pytest.raises(NotImplementedError, match="conduction_speed"):
        exp.render("tvboptim")


@pytest.mark.parametrize("form", FORMS)
def test_a_declared_delay_on_a_network_built_anew_for_each_trial_is_refused(form):
    weights, steps = _connectome()
    exp = _experiment(weights, steps * SPEED * DT, realisation="trial", **_declared(form, steps)[0])
    with pytest.raises(NotImplementedError, match="realisation: trial"):
        exp.render("tvboptim")


def test_a_network_built_anew_for_each_trial_that_declares_no_delay_renders():
    weights, steps = _connectome()
    assert _experiment(weights, steps * SPEED * DT, realisation="trial").render("tvboptim")


@pytest.mark.parametrize("asked", [{"realisation": "trial"}, {"explorations": SPEED_AXIS}])
def test_a_declared_delay_that_no_coupling_reads_refuses_nothing(asked):
    weights, steps = _connectome()
    exp = _experiment(weights, steps * SPEED * DT, delay=steps * DT, **asked)
    for coupling in exp.network.coupling.values():
        coupling.delayed = False
    assert exp.render("tvboptim")


def _interpolating(**declaration):
    """The network of `_connectome` with tract lengths and *declaration*, its coupling reading between two stored steps of the history."""
    weights, steps = _connectome()
    exp = _experiment(weights, steps * SPEED * DT, **declaration)
    for coupling in exp.network.coupling.values():
        coupling.interpolate_delays = True
    return exp


def test_an_interpolated_delay_over_a_function_on_the_delay_is_refused():
    with pytest.raises(NotImplementedError, match="interpolate_delays"):
        _interpolating(transforms=[{"name": "delay", "equation": {"rhs": HALF_UP}}]).render("tvboptim")


def test_an_interpolated_delay_reads_a_stored_delay_layer():
    weights, steps = _connectome()
    other = _declared("a stored delay layer", steps)[1]
    np.testing.assert_allclose(_run(_interpolating(delay=other * DT)), _euler(weights, other), rtol=0, atol=1e-9)


def test_an_interpolated_delay_over_a_function_is_refused_where_a_graph_is_built_outside_a_render():
    from tvbo.adapters.tvboptim import to_tvboptim

    function = _interpolating(transforms=[{"name": "delay", "equation": {"rhs": HALF_UP}}])
    with pytest.raises(NotImplementedError, match="interpolate_delays"):
        to_tvboptim(function.network, return_type="graph", interpolate_delays=True, dt=DT)
    _, steps = _connectome()
    other = _declared("a stored delay layer", steps)[1]
    graph = to_tvboptim(_interpolating(delay=other * DT).network, return_type="graph", interpolate_delays=True, dt=DT)
    np.testing.assert_allclose(np.asarray(graph.delays), other * DT, rtol=0, atol=1e-6)


def test_a_function_that_names_the_integration_step_is_refused_without_one():
    weights, steps = _connectome()
    network = _experiment(weights, steps * SPEED * DT, transforms=[{"name": "delay", "equation": {"rhs": HALF_UP}}]).network
    with pytest.raises(ValueError, match="integration step"):
        network.calculate_delays()
    with pytest.raises(ValueError, match="integration step"):
        network.matrix("delay")
    shorter = (np.maximum(steps, 1) - 1) * DT
    np.testing.assert_allclose(np.asarray(network.calculate_delays(dt=DT)), shorter, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(network.matrix("delay", dt=DT)), shorter, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(network.matrix("delay", apply_transforms=False)), steps * DT, rtol=0, atol=1e-12)


def test_a_delay_layer_declared_without_values_is_refused():
    weights, steps = _connectome()
    exp = _experiment(weights, steps * SPEED * DT, edges=[{"label": "delay", "unit": "ms"}])
    with pytest.raises(ValueError, match="carries no values"):
        exp.network.calculate_delays()
    with pytest.raises(ValueError, match="carries no values"):
        exp.render("tvboptim")


@pytest.mark.parametrize("spelling", ["delay", "delays"])
def test_the_delay_answers_to_either_spelling(spelling):
    weights, steps = _connectome()
    other = _declared("a stored delay layer", steps)[1]
    exp = _experiment(weights, steps * SPEED * DT, transforms=[{"name": spelling, "equation": {"rhs": f"{spelling} * 2"}}])
    exp.network.set_matrix(spelling, other * DT)
    assert exp.network.declares_delays() and exp.network.carries("delay") and exp.network.carries("delays")
    for asked in ("delay", "delays"):
        np.testing.assert_allclose(np.asarray(exp.network.matrix(asked)), 2 * other * DT, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_run(exp), _euler(weights, 2 * other), rtol=0, atol=1e-12)


def test_a_stored_delay_layer_converts_to_the_unit_asked_for():
    weights, steps = _connectome()
    other = _declared("a stored delay layer", steps)[1]
    exp = _experiment(weights, steps * SPEED * DT, delay=other * DT)
    np.testing.assert_allclose(np.asarray(exp.network.calculate_delays(output_unit="s")), other * DT / 1000.0, rtol=1e-12)


def _seconds(weights, lengths=None, **kwargs):
    """`_experiment` on an integration clock of seconds: step, duration and the node's time constant are a thousandth of their millisecond numbers, so the run takes the steps of the millisecond one."""
    spec = copy.deepcopy(SPEC)
    spec["integration"].update({"time_unit": "s", "step_size": DT / 1000.0, "duration": N_STEPS * DT / 1000.0})
    spec["dynamics"]["parameters"]["tau"]["value"] = TAU / 1000.0
    return _experiment(weights, lengths, spec=spec, **kwargs)


def _in_another_unit(case):
    """An experiment whose network states its delays in a unit that is not the integration's clock, and the whole steps each connection is delayed by."""
    weights, steps = _connectome()
    other = _declared("a stored delay layer", steps)[1]
    lengths, in_ms = steps * SPEED * DT, [{"label": "delay", "unit": "ms"}]
    half_up = [{"name": "delay", "equation": {"rhs": HALF_UP}}]
    if case == "a layer in ms on a network in seconds":
        return _experiment(weights, lengths, delay=other * DT, edges=in_ms, time_unit="s"), other
    if case == "a layer in the seconds of its network":
        return _experiment(weights, lengths, delay=other * DT / 1000.0, time_unit="s"), other
    if case == "a layer in ms on a clock of seconds":
        return _seconds(weights, lengths, delay=other * DT, edges=in_ms), other
    if case == "a layer without a unit on a clock of seconds":
        return _seconds(weights, lengths, delay=other * DT / 1000.0), other
    if case == "a function over tract lengths on a clock of seconds":
        return _seconds(weights, lengths, transforms=half_up), np.maximum(steps, 1) - 1
    return _experiment(weights, edge_delay=steps * DT / 1000.0, transforms=half_up, time_unit="s"), np.maximum(steps, 1) - 1


UNIT_CASES = (
    "a layer in ms on a network in seconds",
    "a layer in the seconds of its network",
    "a layer in ms on a clock of seconds",
    "a layer without a unit on a clock of seconds",
    "a function over tract lengths on a clock of seconds",
    "a function over per-edge delays in the seconds of their network",
)


@pytest.mark.parametrize("case", UNIT_CASES)
def test_a_delay_stated_in_another_unit_is_integrated_on_the_integrations_clock(case):
    exp, expected = _in_another_unit(case)
    weights, _ = _connectome()
    assert exp.max_delay == pytest.approx(expected.max() * float(exp.integration.step_size))
    run = _run(exp)
    assert np.abs(run - _euler(weights, np.zeros_like(expected))).max() > 1e-3, "the run is the undelayed network's"
    np.testing.assert_allclose(run, _euler(weights, expected), rtol=0, atol=1e-9)


@pytest.mark.parametrize("home", ["a network in seconds", "a network that declares no time unit"])
@pytest.mark.parametrize("source", ["a stored delay layer", "per-edge delays", "tract lengths"])
def test_the_delays_come_back_in_the_unit_asked_for_whatever_their_source(source, home):
    weights, steps = _connectome()
    connected = weights != 0
    stated, declared = (steps * DT / 1000.0, {"time_unit": "s"}) if home == "a network in seconds" else (steps * DT, {})
    if source == "a stored delay layer":
        network = _experiment(weights, delay=stated, **declared).network
    elif source == "per-edge delays":
        network = _experiment(weights, edge_delay=stated, **declared).network
    else:
        network = _experiment(weights, steps * SPEED * DT, **declared).network
    for unit, per_ms in (("ms", 1.0), ("s", 1e-3)):
        delays = np.asarray(network.calculate_delays(output_unit=unit))
        np.testing.assert_allclose(delays[connected], steps[connected] * DT * per_ms, rtol=1e-12, err_msg=unit)


def test_a_graph_built_outside_a_render_is_built_on_the_clock_it_is_given():
    from tvbo.adapters.tvboptim import _build_graph

    exp, expected = _in_another_unit("a layer in ms on a network in seconds")
    graph = _build_graph(exp.network, dt=DT, time_unit="ms")
    np.testing.assert_allclose(np.asarray(graph.delays), expected * DT, rtol=0, atol=1e-6)


def test_a_network_names_the_source_its_delays_are_read_from():
    weights, steps = _connectome()
    doubled = [{"name": "delay", "equation": {"rhs": "delay * 2"}}]
    assert _experiment(weights, steps * SPEED * DT, delay=steps * DT).network.delay_source() == "layer"
    assert _experiment(weights, edge_delay=steps * DT).network.delay_source() == "edges"
    assert _experiment(weights, steps * SPEED * DT).network.delay_source() == "lengths"
    assert _experiment(weights, steps * SPEED * DT, transforms=doubled).network.delay_source() == "lengths"
    assert _experiment(weights).network.delay_source() is None
    stating = (_experiment(weights, delay=steps * DT), _experiment(weights, edge_delay=steps * DT), _declaring(FORMS[1]))
    assert all(exp.network.states_delays() for exp in stating)
    assert not _experiment(weights, steps * SPEED * DT).network.states_delays()


def test_a_network_reports_whether_and_how_it_declares_its_delays():
    weights, steps = _connectome()
    assert _experiment(weights, steps * SPEED * DT).network.delay_declaration() is None
    assert not _experiment(weights, edge_delay=steps * DT).network.declares_delays()
    assert _declaring(FORMS[0]).network.delay_declaration() == "a `delay` layer"
    assert _declaring(FORMS[1]).network.delay_declaration() == "a transform on `delay`"
    both = _experiment(weights, delay=steps * DT, transforms=[{"name": "delay", "equation": {"rhs": "delay * 2"}}])
    assert both.network.delay_declaration() == "a `delay` layer and a transform on `delay`"
    assert not _experiment(weights, steps * SPEED * DT).network.carries("delay"), "tract lengths are carried as length"


def test_a_delay_derived_from_tract_lengths_is_not_materialised_as_a_layer():
    weights, steps = _connectome()
    with pytest.raises(KeyError, match="delay"):
        _experiment(weights, steps * SPEED * DT).network.materialize("delay")
    stored = _experiment(weights, steps * SPEED * DT, delay=steps * DT).network.materialize("delay")
    assert stored.declares_delays()


def test_the_report_states_the_delays_a_run_integrates():
    from tvbo.utils.report import experiment_facts

    weights, steps = _connectome()
    assert experiment_facts(_experiment(weights, steps * SPEED * DT))["Delays"] == "2 mm/ms"
    assert experiment_facts(_declaring(FORMS[0]))["Delays"] == "a `delay` layer"
    assert experiment_facts(_declaring(FORMS[1]))["Delays"] == "2 mm/ms, then a transform on `delay`"
    assert experiment_facts(_declaring(FORMS[1], listed_edges=True))["Delays"] == "1 ms-6 ms, then a transform on `delay`"


ABSENT = {
    "a misspelt weight": {"name": "wieght", "equation": {"rhs": "wieght / 2"}},
    "a length on a network without tract lengths": {"name": "length", "equation": {"rhs": "length / 2"}},
    "a delay on a network without lengths or delays": {"name": "delay", "equation": {"rhs": "delay * 2"}},
    "a matrix the network does not hold": {"name": "fc", "equation": {"rhs": "fc * 2"}},
}
"""A transform whose target names no layer of a network that holds weights only."""


@pytest.mark.parametrize("target", ABSENT)
def test_a_transform_on_a_layer_the_network_lacks_is_refused_at_read(target):
    weights, _ = _connectome()
    exp = _experiment(weights, transforms=[ABSENT[target]])
    for coupling in exp.network.coupling.values():
        coupling.delayed = False
    with pytest.raises(ValueError, match="has no layer of that name"):
        exp.network.matrix("weight")
    with pytest.raises(ValueError, match="has no layer of that name"):
        exp.run(format="tvboptim")


def test_a_transform_on_a_layer_the_network_holds_is_read():
    weights, steps = _connectome()
    exp = _experiment(weights, steps * SPEED * DT, transforms=[{"name": "fc", "equation": {"rhs": "fc * 2"}}])
    exp.network.set_matrix("fc", weights)
    np.testing.assert_allclose(np.asarray(exp.network.matrix("fc")), 2 * weights, rtol=0, atol=1e-12)
    declared = _experiment(weights, transforms=[{"name": "fc", "equation": {"rhs": "fc * 2"}}], edges=[{"label": "fc"}])
    declared.network.refuse_transforms_without_a_layer()


def test_a_generated_network_is_not_asked_for_its_layers_before_it_is_built():
    from tvbo import Network

    ring = Network(
        number_of_nodes=8,
        graph_generator={"name": "ring", "type": "Cycle"},
        transforms=[{"name": "length", "equation": {"rhs": "length * 2"}}],
    )
    ring.refuse_transforms_without_a_layer()


def test_a_graph_built_outside_a_render_holds_the_declared_delays():
    from tvbo.adapters.tvboptim import _build_graph

    weights, steps = _connectome()
    for form in FORMS:
        declaration, declared_steps = _declared(form, steps)
        graph = _build_graph(_experiment(weights, steps * SPEED * DT, **declaration).network, dt=DT)
        np.testing.assert_allclose(np.asarray(graph.delays), declared_steps * DT, rtol=0, atol=1e-6, err_msg=form)


def test_rendered_code_refuses_to_build_a_declared_network_without_its_delays():
    weights, _ = _connectome()
    module = _declaring(FORMS[0]).execute("tvboptim")
    with pytest.raises(ValueError, match="create_network: this network states its delays itself"):
        module.create_network(weights)
    with pytest.raises(ValueError, match="run_experiment: this network states its delays itself"):
        module.run_experiment(weights)


OSCILLATORS = {
    "dynamics": {"name": "Generic2dOscillator", "iri": "tvbo:Generic2dOscillator"},
    "network": {
        "number_of_nodes": N,
        "nodes": [{"id": i} for i in range(N)],
        "parameters": {"conduction_speed": {"name": "conduction_speed", "value": SPEED, "unit": "mm_per_ms"}},
        "coupling": {
            "Linear": {
                "name": "Linear",
                "iri": "tvbo:Linear",
                "delayed": True,
                "parameters": {"a": {"name": "a", "value": 0.3}, "b": {"name": "b", "value": 0.05}},
            }
        },
    },
    "integration": {"method": "Heun", "step_size": 0.25, "duration": 20.0},
}
"""Curated oscillators under the curated linear coupling declared delayed, which the Python graph runner integrates."""


def _graph_run(lengths, step, **declaration):
    """The Python graph runner's trace of `OSCILLATORS` over the weights of `_connectome`, tract *lengths* and *declaration*, as ``(time, variable × node)``."""
    spec = {**OSCILLATORS, "integration": {**OSCILLATORS["integration"], "step_size": step}}
    data = _experiment(_connectome()[0], lengths, spec=spec, **declaration).run(format="python").integration.data
    return np.asarray(data).reshape(data.shape[0], -1)


@pytest.mark.parametrize("form", FORMS)
def test_the_python_graph_runner_integrates_the_declared_delays(form):
    step = 0.25
    _, steps = _connectome()
    declaration, declared_steps = _declared(form, steps)
    if "delay" in declaration:
        declaration["delay"] = declared_steps * step
    undeclared = _graph_run(steps * SPEED * step, step)
    run = _graph_run(steps * SPEED * step, step, **declaration)
    assert np.abs(run - undeclared).max() > 1e-6, f"with {form} the run is the one of tract length over speed"
    np.testing.assert_allclose(run, _graph_run(declared_steps * SPEED * step, step), rtol=0, atol=1e-12)


def test_the_python_graph_runner_reads_a_delay_layer_in_the_unit_it_states():
    step = 0.25
    _, steps = _connectome()
    other = _declared("a stored delay layer", steps)[1]
    in_seconds = {"delay": other * step / 1000.0, "edges": [{"label": "delay", "unit": "s"}]}
    run = _graph_run(steps * SPEED * step, step, **in_seconds)
    np.testing.assert_allclose(run, _graph_run(other * SPEED * step, step), rtol=0, atol=1e-12)


def _pyrates_run(exp, **kwargs):
    """The PyRates trace of *exp* as ``(time, node)``."""
    data = exp.run(format="pyrates", **kwargs).integration.data
    return np.asarray(data).reshape(len(data), -1)


def test_pyrates_integrates_a_stored_delay_layer():
    pytest.importorskip("pyrates")
    weights, steps = _connectome()
    other = _declared("a stored delay layer", steps)[1]
    run = _pyrates_run(_experiment(weights, steps * SPEED * DT, delay=other * DT))

    def by_length(delays):
        return _pyrates_run(_experiment(weights, delays * SPEED * DT), matrix_edge_threshold=0)

    assert np.abs(run - by_length(steps)).max() > 1e-3, "the run is the one of tract length over speed"
    np.testing.assert_allclose(run, by_length(other), rtol=0, atol=1e-12)


LANES = {
    "functions": {"node_mean": {"source_code": "jnp.mean(data[:, 0, :], axis=0)", "arguments": {"data": {}}}},
    "events": {
        "drive": {
            "event_type": "stimulus",
            "equation": {"rhs": "amplitude"},
            "parameters": {"amplitude": {"shape": "(n_nodes,)", "value": 0.0}},
            "target_variable": "x",
        }
    },
    "observations": {
        "level": {
            "source": ["x"],
            "dims": ["node"],
            "pipeline": [{"function": "node_mean", "arguments": {"data": {"value": "integration.result"}}}],
        }
    },
}
"""A per-node drive and the time mean of every node, which `tvbo.run.lanes` runs once per row of drive amplitudes."""


def _lane(weights, lengths=None, **carrier):
    """The time mean of every node in the one lane that drives node 0 alone, over *weights* with the delays as given."""
    import xarray as xr

    from tvbo.run import lanes

    exp = _experiment(weights, lengths, spec={**SPEC, **LANES}, **carrier)
    drive = xr.Dataset({"x.amplitude": (("programme", "node"), np.eye(N)[:1])}, coords={"node": [f"r{i}" for i in range(N)]})
    return np.asarray(lanes.run_lanes(exp, drive, baseline=False)["level"].values[0])


@pytest.mark.parametrize("carrier", ["per-edge delays", "a delay layer in seconds", *FORMS])
def test_a_lane_integrates_the_delays_of_its_network(carrier):
    weights, steps = _connectome()
    if carrier == "per-edge delays":
        given, lane = steps, _lane(weights, edge_delay=steps * DT)
    elif carrier == "a delay layer in seconds":
        given = _declared("a stored delay layer", steps)[1]
        lane = _lane(weights, steps * SPEED * DT, delay=given * DT / 1000.0, edges=[{"label": "delay", "unit": "s"}])
    else:
        declaration, given = _declared(carrier, steps)
        lane = _lane(weights, steps * SPEED * DT, **declaration)
    assert np.abs(lane - _lane(weights, np.zeros_like(steps) * DT)).max() > 1e-4, "the lane is the undelayed network's"
    np.testing.assert_allclose(lane, _lane(weights, given * SPEED * DT), rtol=1e-6)


def test_a_lane_integrates_the_tract_lengths_its_network_gives():
    weights, steps = _connectome()
    lane = _lane(weights, steps * SPEED * DT, transforms=[{"name": "length", "equation": {"rhs": "length * 2"}}])
    assert np.abs(lane - _lane(weights, steps * SPEED * DT)).max() > 1e-4, "the lane is the one of the lengths as stored"
    np.testing.assert_allclose(lane, _lane(weights, 2 * steps * SPEED * DT), rtol=1e-6)
