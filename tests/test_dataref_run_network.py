"""A reference reads the network and the parameter values a run used, from what the run wrote beside its result container.

A multi-node run freezes its network beside its container (``<stem>_network.yaml`` and ``.h5``, :func:`tvbo.data.dataref.network_companion_path`) and its recipe in the container's provenance sidecar (``<stem>.yaml``). ``output: network.<name>`` reads the first and ``output: dynamics.<parameter>`` the second, so a figure or an analysis reads what the run used and never a rebuild of it.

Self-contained: each run is a small synthetic container, a recipe written as a run's sidecar records one, and a network saved by the writer a run freezes its network with (``Network.save``).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import xarray as xr
import yaml

import tvbo.datamodel.pydantic as P
from tvbo import Network
from tvbo.adapters import bsplot
from tvbo.data import analysis_io, dataref
from tvbo.data.param_io import resolve_network_node
from tvbo.datamodel import Analysis, Coordinate, DataRef, Node, Parameter
from tvbo.utils.study_layout import study_path

LABELS = ["A", "B", "C", "D"]
WEIGHTS = np.array(
    [
        [0.0, 1.0, 2.0, 0.0],
        [0.5, 0.0, 0.0, 3.0],
        [0.0, 0.25, 0.0, 1.0],
        [4.0, 0.0, 0.5, 0.0],
    ]
)
"""Directed weights, so the row sums (in-strength) and the column sums (out-strength) differ."""
POSITIONS = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0], [2.5, 2.5, 0.0]])
"""A 2 x 2 sheet in mm, every coordinate exact in binary."""
TEMPLATE = np.array([-1.0, -0.5, 0.5, 1.0])
"""A per-node attribute, carried as a network builder's node parameters arrive."""
PARAMETERS = {
    "omega_mean_hz": {"name": "omega_mean_hz", "value": 10.0},
    "K": {"name": "K", "value": 0.01},
    "gradient_template": {"name": "gradient_template", "value": 0.0},
    "noise": {"name": "noise", "value": 0.1, "distribution": {"name": "Uniform", "domain": {"lo": 0.0, "hi": 1.0}}},
}
"""The dynamics parameters the run's recipe declares."""


def _network():
    """The network the run used: four positioned nodes, each carrying ``gradient_template``, on the directed weights."""
    net = Network.from_matrix(weights=WEIGHTS, labels=LABELS)
    net.nodes = [
        Node(
            id=i,
            label=label,
            position=Coordinate(x=x, y=y, z=z),
            parameters={"gradient_template": Parameter(name="gradient_template", value=value)},
        )
        for i, (label, (x, y, z), value) in enumerate(zip(LABELS, POSITIONS.tolist(), TEMPLATE.tolist(), strict=True))
    ]
    return net


def _run(results, data=None, network=True):
    """Experiment 1's result as a run writes it under *results*: the container holding *data*, its recipe sidecar and, with *network*, its network frozen beside it."""
    results.mkdir(parents=True, exist_ok=True)
    container = results / "exp-1_model-M_result.h5"
    if data is None:
        data = xr.Dataset({"theta": (("time", "node"), np.zeros((3, 4)))}, coords={"node": LABELS})
    data.to_netcdf(container, engine="h5netcdf")
    recipe = {"id": 1, "dynamics": {"name": "M", "parameters": PARAMETERS}}
    dataref.sidecar_path(container).write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    if network:
        _network().save(dataref.network_companion_path(container))
    return container


def _read(results, output):
    return dataref.resolve_dataref(DataRef(experiment=1, output=output), results_root=results)


def test_a_runs_weights_positions_and_node_attributes_are_the_ones_it_froze(tmp_path):
    """Each comes from the network beside the container, labelled by its nodes; positions run along ``coordinate`` x, y, z."""
    _run(tmp_path)
    weight = _read(tmp_path, "network.weight")
    assert weight.dims == ("node_i", "node_j") and list(weight["node_i"].values) == LABELS
    np.testing.assert_array_equal(weight.values, WEIGHTS)
    positions = _read(tmp_path, "network.positions")
    assert positions.dims == ("node", "coordinate") and list(positions["coordinate"].values) == ["x", "y", "z"]
    assert list(positions["node"].values) == LABELS
    np.testing.assert_array_equal(positions.values, POSITIONS)
    template = _read(tmp_path, "network.nodes.gradient_template")
    assert template.dims == ("node",) and list(template["node"].values) == LABELS
    np.testing.assert_array_equal(template.values, TEMPLATE)


def test_a_runs_instrength_is_the_networks_own_in_strength_of_those_weights(tmp_path):
    """The network's definition (`resolve_network_node`), the row sums of its weights, not the column sums."""
    _run(tmp_path)
    instrength = _read(tmp_path, "network.instrength")
    assert instrength.dims == ("node",) and list(instrength["node"].values) == LABELS
    np.testing.assert_array_equal(instrength.values, resolve_network_node(_network(), "instrength"))
    assert not np.allclose(instrength.values, WEIGHTS.sum(axis=0)), "the column sums are the out-strength"


def test_dynamics_reads_the_value_the_runs_recipe_declares(tmp_path):
    _run(tmp_path)
    value = _read(tmp_path, "dynamics.omega_mean_hz")
    assert value.dims == () and float(value) == 10.0


@pytest.mark.parametrize(
    ("output", "said"),
    [
        (
            "network.tracts",
            r"named 'tracts' \(edge matrices: \[.*'weight'.*\]; per-node quantities: \[.*'gradient_template'\]\)",
        ),
        (
            "dynamics.omega",
            r"declares no parameter 'omega'; it declares \['K', 'gradient_template', 'noise', 'omega_mean_hz'\]",
        ),
    ],
    ids=["network", "dynamics"],
)
def test_an_unknown_name_is_refused_naming_what_the_run_holds(tmp_path, output, said):
    _run(tmp_path)
    with pytest.raises(KeyError, match=said):
        _read(tmp_path, output)


SWEPT = xr.Dataset({"observation__r": (("M.K", "node"), np.zeros((2, 4)))}, coords={"M.K": [0.01, 0.02], "node": LABELS})
TUNED = xr.Dataset({"estimate__K": (("node",), np.full(4, 0.02))}, coords={"node": LABELS})


@pytest.mark.parametrize(
    ("data", "output", "said"),
    [
        (SWEPT, "dynamics.K", "its values are in `M.K`"),
        (TUNED, "dynamics.K", "its values are in `estimate__K`"),
        (None, "dynamics.gradient_template", "its values are in `network.nodes.gradient_template`"),
        (None, "dynamics.noise", "its values are in its `distribution`"),
    ],
    ids=["swept", "tuned", "per-node", "drawn"],
)
def test_a_parameter_the_run_did_not_hold_at_one_value_is_refused_naming_where_its_values_are(tmp_path, data, output, said):
    """The recipe's value is then not the run's, so returning it would be a plausible wrong number."""
    _run(tmp_path, data=data)
    with pytest.raises(ValueError, match=said):
        _read(tmp_path, output)


def test_a_result_with_no_network_beside_it_is_refused_by_name(tmp_path):
    _run(tmp_path, network=False)
    with pytest.raises(
        ValueError, match=r"exp-1_model-M_result.h5: its run wrote no network beside it \(exp-1_model-M_network.yaml\)"
    ):
        _read(tmp_path, "network.weight")


def test_a_network_that_takes_a_layer_through_a_reference_is_not_read_again(tmp_path):
    """Its companion keeps the reference, and reading the source now would not give what the run read."""
    container = _run(tmp_path, network=False)
    frozen = {
        "nodes": [{"id": i, "label": label} for i, label in enumerate(LABELS)],
        "edges": [{"label": "weight", "used": {"analysis": "gains", "output": "gain"}}],
    }
    dataref.network_companion_path(container).write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="takes the reference to analysis 'gains', output 'gain' when it is bound to a run"):
        _read(tmp_path, "network.positions")


def test_a_figure_layer_reads_the_network_its_run_used(tmp_path):
    """Planned by `build_context` and read by `load_layer`, which the emitted script's layer reader shares (`output_array`)."""
    _run(study_path("results", root=tmp_path))
    bsplot._container_path.cache_clear()
    layer = P.Layer(used=P.DataRef(experiment="1", output="network.instrength"), encoding=P.Encoding(x="node"))
    figure = P.Figure(name="fig", layout="a", panels={"a": P.Panel(panel_key="a", kind="cartesian", layers=[layer])})
    (resolved,) = bsplot.build_context(figure, tmp_path, "out.png")["panels"][0]["layers"]
    np.testing.assert_array_equal(bsplot.load_layer(resolved).values, resolve_network_node(_network(), "instrength"))


@pytest.fixture
def calls(monkeypatch):
    """A throwaway module holding an analysis callable that returns its arguments, importable by bare name."""
    mod = types.ModuleType("_test_run_network_calls")

    def echo(positions=None, mean_frequency=None):
        return {"positions": positions, "mean_frequency": mean_frequency}

    mod.echo = echo
    monkeypatch.setitem(sys.modules, "_test_run_network_calls", mod)
    return mod


def test_an_analysis_argument_reads_the_network_and_the_values_its_run_used(tmp_path, calls):
    """Ordered after experiment 1 by `dependencies`, and handed the arrays `run_analysis` resolves."""
    _run(tmp_path)
    analysis = Analysis(
        name="sheet",
        callable={"name": "echo", "module": "_test_run_network_calls"},
        arguments={
            "positions": {"used": {"experiment": 1, "output": "network.positions"}},
            "mean_frequency": {"used": {"experiment": 1, "output": "dynamics.omega_mean_hz"}},
        },
    )
    assert analysis_io.dependencies(analysis)["experiments"] == {"1"}
    path = analysis_io.run_analysis(analysis, tmp_path)
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        positions = ds["observation__positions"].load()
        assert float(ds["observation__mean_frequency"]) == 10.0
    assert positions.dims == ("node", "coordinate") and list(positions["node"].values) == LABELS
    np.testing.assert_array_equal(positions.values, POSITIONS)
