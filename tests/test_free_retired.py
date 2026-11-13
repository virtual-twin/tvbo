"""A parameter is named where something acts on it, and `Parameter.free` is gone.

An optimization stage fits a parameter at the parameter's own layout and writes `shape` or `heterogeneous` only to change it; a continuation holds a parameter to the target of an algorithm's objective through `held_parameters`; the report lists what acts on each parameter. A record that still carries `free` is refused on every load path, and a stored result sidecar that carries it is still read as the data it is.
"""

from __future__ import annotations

import pytest
import yaml

from tvbo import SimulationExperiment, datamodel
from tvbo.adapters.bifurcationkit import BifurcationKitAdapter
from tvbo.adapters.numcont import NumContAdapter
from tvbo.adapters.pyrates_bifurcation import PyRatesBifurcationAdapter
from tvbo.classes.optimization import fitted_parameters
from tvbo.data import dataref, provenance
from tvbo.utils import keyed_items, parameters_acted_on, report

SPEC = """
id: 1
dynamics:
  name: Leak
  parameters:
    g: {value: 0.0, heterogeneous: true, shape: "(n_nodes,)"}
    k: {value: 1.0}
    tau: {value: 5.0}
  state_variables:
    x: {initial_value: 0.0, equation: {rhs: "(k - g - x) / tau"}, coupling_variable: true}
  output: [x]
network:
  number_of_nodes: 1
  nodes: [{id: 0, label: r0}]
  edges: []
integration: {method: euler, step_size: 0.1, duration: 4.0, unit: ms}
observations:
  mean_x: {source: [x], aggregation: mean}
algorithms:
  tune:
    objective: {type: activity_target, target_variable: x, target_value: 0.5}
    observations: [mean_x]
    update_rules:
      - {name: g_update, target_parameter: {name: g}, equation: {rhs: "g + eta * (mean_x - 0.5)"}}
    hyperparameters: [{name: eta, value: 0.3}]
    n_iterations: 2
    simulation_period: 2.0
optimizations:
  fit:
    loss: {function: mse, arguments: {x: {value: observations.mean_x.data}}}
    free_parameters: [ENTRY]
    algorithm: adam
    learning_rate: 0.01
    max_iterations: 2
explorations:
  sweep:
    space:
      - {parameter: Leak.tau, explored_values: [4.0, 5.0]}
continuations:
  eq:
    dynamics: Leak
    free_parameters: [{name: k, domain: {lo: 0.0, hi: 2.0}}]
    held_parameters: [HELD]
"""


def _experiment(entry="{parameter: Leak.k}", held="{parameter: g, algorithm: tune}", spec=SPEC):
    return SimulationExperiment.from_string(spec.replace("ENTRY", entry).replace("HELD", held))


def _continuation(experiment):
    ((_, continuation),) = keyed_items(experiment.continuations, "continuations")
    return continuation


@pytest.mark.parametrize(
    "entry, layout",
    [
        ("{parameter: Leak.g}", (True, "(n_nodes,)")),
        ('{parameter: Leak.g, shape: "()"}', (False, None)),
        ("{parameter: Leak.k}", (False, None)),
        ("{parameter: Leak.g, heterogeneous: false}", (False, None)),
        ("{parameter: Leak.k, heterogeneous: true}", (True, None)),
    ],
    ids=["inherits-per-node", "narrows-to-scalar", "inherits-scalar", "false-narrows-to-scalar", "widens-to-per-node"],
)
def test_a_stage_entry_fits_its_parameter_at_the_parameters_own_layout(entry, layout):
    experiment = _experiment(entry)
    (fitted,) = fitted_parameters(experiment.optimizations["fit"], experiment)
    assert (fitted["heterogeneous"], fitted["shape"]) == layout


@pytest.mark.parametrize(
    "entry, name, layout",
    [
        ("{parameter: Leak.g, heterogeneous: true}", "g", r"\(n_nodes,\)"),
        ('{parameter: Leak.g, shape: "(n_nodes,)"}', "g", r"\(n_nodes,\)"),
        ('{parameter: Leak.k, shape: "()"}', "k", "of a scalar"),
        ("{parameter: Leak.k, heterogeneous: false}", "k", "of a scalar"),
    ],
    ids=["heterogeneous", "shape", "scalar", "false-on-a-scalar"],
)
def test_an_entry_that_restates_its_parameters_layout_is_refused_naming_it(entry, name, layout):
    experiment = _experiment(entry)
    with pytest.raises(
        ValueError,
        match=rf"optimization stage 'fit': the entry for '{name}' writes the layout {layout} that '{name}' declares",
    ):
        fitted_parameters(experiment.optimizations["fit"], experiment)


@pytest.mark.backend_tvboptim
def test_a_restatement_is_refused_when_the_fit_is_rendered():
    pytest.importorskip("tvboptim")
    with pytest.raises(ValueError, match="the entry for 'g' writes the layout"):
        _experiment("{parameter: Leak.g, heterogeneous: true}").render_code("tvboptim")


def test_a_held_parameter_is_held_to_the_target_of_its_algorithms_objective():
    experiment = _experiment()
    assert BifurcationKitAdapter(experiment)._derive_constraints(_continuation(experiment)) == [
        {"parameter": "g", "target_variable": "x", "target_value": 0.5}
    ]


@pytest.mark.parametrize(
    "held, spec, message",
    [
        (
            "{parameter: g, algorithm: fic}",
            SPEC,
            "holds 'g' by algorithm 'fic', which the experiment does not declare; its algorithms are tune",
        ),
        (
            "{parameter: g, algorithm: tune}",
            SPEC.replace(", target_value: 0.5}", "}"),
            "whose objective states no target_variable and target_value to hold it to",
        ),
        (
            "{parameter: k, algorithm: tune}",
            SPEC,
            "holds 'k' by algorithm 'tune', none of whose update rules targets it; they target g",
        ),
    ],
    ids=["unknown-algorithm", "no-target", "not-tuned"],
)
def test_a_held_parameter_its_algorithm_does_not_hold_is_refused(held, spec, message):
    experiment = _experiment(held=held, spec=spec)
    with pytest.raises(ValueError, match=message):
        BifurcationKitAdapter(experiment)._derive_constraints(_continuation(experiment))


@pytest.mark.parametrize("adapter", [NumContAdapter, PyRatesBifurcationAdapter], ids=["numcont", "pyrates-bifurcation"])
def test_a_backend_that_holds_no_parameter_refuses_a_held_one(adapter):
    experiment = _experiment()
    with pytest.raises(
        ValueError,
        match=rf"holds 'g' to a target, and {adapter.__name__} continues no held parameter; continue it on bifurcationkit",
    ):
        adapter.refuse_unrunnable(experiment.dynamics, _continuation(experiment), "eq")


def test_what_acts_on_each_parameter_is_listed_in_the_report():
    experiment = _experiment()
    assert parameters_acted_on(experiment, experiment.dynamics) == {
        "g": "tuned by tune; held by eq",
        "k": "fitted by fit; continued by eq",
        "tau": "swept by sweep",
    }
    text = experiment.report()
    assert (
        "**Parameters acted on**\n\n- $g$: tuned by tune; held by eq\n- $k$: fitted by fit; continued by eq\n- $\\tau$: swept by sweep\n"
        in text
    )


FREE = SPEC.replace("    k: {value: 1.0}\n", "    k: {value: 1.0, free: true}\n")
FREE_DERIVED = SPEC.replace(
    "  state_variables:\n", '  derived_parameters:\n    twice_k: {equation: {rhs: "2 * k"}, free: true}\n  state_variables:\n'
)


@pytest.mark.parametrize("cls, spec", [("Parameter", FREE), ("DerivedParameter", FREE_DERIVED)])
def test_free_is_refused_on_every_load_path(tmp_path, cls, spec):
    message = rf"{cls}\.free was retired in tvbo 0\.9\.0: name the parameter where it is acted on"
    with pytest.raises(ValueError, match=message):
        getattr(datamodel, cls)(name="k", free=True)
    with pytest.raises(ValueError, match=message):
        getattr(datamodel, cls).model_validate({"name": "k", "free": True})
    with pytest.raises(ValueError, match=message):
        _experiment(spec=spec)
    path = tmp_path / "exp.yaml"
    path.write_text(spec.replace("ENTRY", "{parameter: Leak.k}").replace("HELD", "{parameter: g, algorithm: tune}"))
    with pytest.raises(ValueError, match=message):
        SimulationExperiment.from_file(path)


def test_a_stored_sidecar_that_carries_free_is_read_as_data(tmp_path, monkeypatch):
    container = tmp_path / "exp-1_result.h5"
    container.write_bytes(b"")
    sidecar = dataref.sidecar_path(container)
    sidecar.write_text(
        yaml.safe_dump(
            {
                "id": 1,
                "dynamics": {"parameters": {"k": {"value": 1.0, "free": True}}},
                "algorithms": {"tune": {"n_iterations": 2}},
            }
        )
    )
    assert dataref.source_producers(container) == ["tune"]
    monkeypatch.setattr(dataref, "locate_exp_container", lambda out_dir, experiment: container)
    assert report.sidecar_value(report.result_sidecar(tmp_path, 1), "dynamics", "parameters", "k", "free") is True
    provenance.emit(container=container, produced_by="tvbo:experiment/1")
    written = yaml.safe_load(sidecar.read_text())
    assert written["dynamics"]["parameters"]["k"]["free"] is True and "provenance" in written
