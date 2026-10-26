"""The Python graph runner couples its nodes as tvboptim does: a node without afferents still receives the coupling's post-transform of zero, and a coupling reads the states it names, wherever the model declares them.

tvboptim is the reference at every recorded time. Linear's ``b`` is what a source-only node receives, so dropping it moves that node off tvboptim's trajectory. JansenRit transmits ``y1`` and ``y2``, its second and third states, so a pre-transform that read the source's whole state vector would take ``x_j[0]`` as ``y0``; SigmoidalJansenRit is run both as curated (``x_j[0] - x_j[1]``) and with the states named (``y1_j - y2_j``), and tvboptim integrates the named form. Every run starts from the model's initial values.
"""

import numpy as np
import pytest
import xarray as xr

from tvbo import SimulationExperiment

W = np.array([[0.0, 1.0, 0.0, 4.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 5.0], [0.0, 6.0, 0.0, 0.5]]) / 4.0
"""Source-by-target: ``W[i, j]`` is the weight of the edge ``i → j``. Column 0 is empty, so node 0 has no afferents."""

G2D = {"name": "Generic2dOscillator", "iri": "tvbo:Generic2dOscillator"}
JR = {"name": "JansenRit", "iri": "tvbo:JansenRit"}


def _parameters(values):
    return {name: {"name": name, "value": value} for name, value in values.items()}


LINEAR = {"name": "Linear", "iri": "tvbo:Linear", "delayed": False, "parameters": _parameters({"a": 0.3, "b": 0.05})}
SJR_PARAMETERS = _parameters({"a": 50.0, "midpoint": 6.0, "cmax": 0.005, "r": 0.56, "cmin": 0.0})
SJR_NAMED = {
    "name": "SJRNamed",
    "delayed": False,
    "incoming_states": ["y1", "y2"],
    "parameters": SJR_PARAMETERS,
    "pre_expression": {"rhs": "cmin + (cmax - cmin)/(exp(r*(midpoint - (y1_j - y2_j))) + 1.0)"},
    "post_expression": {"rhs": "a*gx"},
}
SJR_CURATED = {"name": "SigmoidalJansenRit", "iri": "tvbo:SigmoidalJansenRit", "delayed": False, "parameters": SJR_PARAMETERS}


def _experiment(dynamics, coupling, method="Heun", scale=1.0):
    edges = [
        {
            "source": i,
            "target": j,
            "directed": True,
            "parameters": {"weight": {"name": "weight", "value": float(W[i, j] * scale)}},
        }
        for i in range(4)
        for j in range(4)
        if W[i, j]
    ]
    network = {
        "number_of_nodes": 4,
        "nodes": [{"id": k} for k in range(4)],
        "edges": edges,
        "coupling": {coupling["name"]: coupling},
    }
    integration = {"method": method, "step_size": 0.05, "duration": 10.0}
    return SimulationExperiment(dynamics=dynamics, network=network, integration=integration)


def _by_time(result):
    """*result*'s trace as ``{time: (variable, node) states}``."""
    data = next(
        d
        for c in (result, getattr(result, "integration", None))
        for d in (c, getattr(c, "data", None))
        if isinstance(d, xr.DataArray)
    )
    if "mode" in data.dims:
        data = data.isel(mode=0)
    data = data.transpose("time", "variable", "node")
    times = np.round(np.asarray(data["time"].values, dtype=float), 9)
    return dict(zip(times, np.asarray(data.values), strict=True))


def _gap(a, b, node=slice(None)):
    """The largest |a - b| over the times both record, at *node*, and how many times they share."""
    common = sorted(set(a) & set(b))
    return max(float(np.max(np.abs(a[t][:, node] - b[t][:, node]))) for t in common), len(common)


@pytest.mark.backend_tvboptim
@pytest.mark.parametrize("method", ["Euler", "Heun"])
def test_a_node_without_afferents_receives_the_post_transform_of_zero(method):
    """Node 0 has no afferents, so its input is Linear's ``a·0 + b``: the runner follows tvboptim there, and a zero input would not."""
    pytest.importorskip("tvboptim")
    python = _by_time(_experiment(G2D, LINEAR, method).run(format="python"))
    tvboptim = _by_time(_experiment(G2D, LINEAR, method).run(format="tvboptim"))
    without_b = _by_time(
        _experiment(G2D, {**LINEAR, "parameters": _parameters({"a": 0.3, "b": 0.0})}, method).run(format="tvboptim")
    )

    gap, shared = _gap(python, tvboptim)
    assert shared == 200 and gap < 1e-12
    assert _gap(python, without_b, node=0)[0] > 1e-3


@pytest.mark.backend_tvboptim
@pytest.mark.parametrize("coupling", [SJR_NAMED, SJR_CURATED], ids=["named", "curated"])
def test_the_coupling_reads_the_states_it_names(coupling):
    """JansenRit's coupled states are its second and third, and the runner follows tvboptim's named SigmoidalJansenRit whichever way the pre-transform names them; without the coupling it would not."""
    pytest.importorskip("tvboptim")
    python = _by_time(_experiment(JR, coupling).run(format="python"))
    tvboptim = _by_time(_experiment(JR, SJR_NAMED).run(format="tvboptim"))
    uncoupled = _by_time(_experiment(JR, SJR_NAMED, scale=0.0).run(format="tvboptim"))

    gap, shared = _gap(python, tvboptim)
    assert shared == 200 and gap < 1e-12
    assert _gap(python, uncoupled)[0] > 1e-3
