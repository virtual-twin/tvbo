"""A Jansen-Rit network coupled through the curated SigmoidalJansenRit reproduces TVB's own implementation on every backend.

The reference is TVB's built-in ``coupling.SigmoidalJansenRit`` and ``models.JansenRit``, not anything tvbo generates. The coupling reads ``x_j[0] - x_j[1]``, the source's ``y1 - y2``: a printer that drops the component index turns it into ``midpoint`` and the coupling into a constant, which the gain of 10 makes visible within the 100 ms run. The network is declared by its edges, and the transformed variant normalises it, so the connectome and its transform must both reach the solver.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tvb")

from tvbo.classes.experiment import SimulationExperiment

GAIN = 10.0

_SPEC = """
id: 1
dynamics:
  name: JansenRit
  iri: tvbo:JansenRit
network:
  number_of_nodes: 3
  nodes:
    - {id: 0, label: a}
    - {id: 1, label: b}
    - {id: 2, label: c}
  edges:
    - {source: 0, target: 1, weight: 1.0, directed: true}
    - {source: 1, target: 2, weight: 0.5, directed: true}
    - {source: 2, target: 0, weight: 2.0, directed: true}
%(transforms)s  coupling:
    c_glob:
      iri: tvbo:SigmoidalJansenRit
      delayed: false
      parameters: {a: {value: %(gain)s}}
integration: {method: Euler, step_size: 0.1, duration: 100.0}
"""

_TRANSFORMS = {
    "raw": "",
    "normalised": '  transforms:\n    - name: weight\n      equation: {rhs: "weight / max(weight)"}\n',
}

EDGES = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
"""The declared edges as TVB reads them, target by source."""

WEIGHTS = {"raw": EDGES, "normalised": EDGES / EDGES.max()}


def _experiment(tmp_path, variant):
    spec = tmp_path / f"jr_sjr_{variant}.yaml"
    spec.write_text(_SPEC % {"transforms": _TRANSFORMS[variant], "gain": GAIN})
    return SimulationExperiment.from_file(str(spec))


def _tvb_reference(weights, duration):
    """TVB's own Jansen-Rit network with its built-in SigmoidalJansenRit, from tvbo's uniform 0.1 initial state; returns times and ``(time, state, node)``."""
    from tvb.datatypes import connectivity
    from tvb.simulator import coupling, integrators, models, monitors, simulator

    n = weights.shape[0]
    conn = connectivity.Connectivity(
        weights=weights,
        tract_lengths=np.zeros((n, n)),
        number_of_regions=n,
        region_labels=np.array([f"r{i}" for i in range(n)]),
        centres=np.zeros((n, 3)),
    )
    conn.speed = np.array([4.0])
    model = models.JansenRit(variables_of_interest=models.JansenRit.state_variables)
    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=coupling.SigmoidalJansenRit(a=np.array([GAIN])),
        integrator=integrators.EulerDeterministic(dt=0.1),
        monitors=[monitors.Raw()],
        initial_conditions=np.full((1, len(model.state_variables), n, 1), 0.1),
        simulation_length=duration,
    )
    sim.configure()
    ((times, data),) = sim.run()
    return times, data[..., 0], list(model.state_variables)


@pytest.mark.parametrize("variant", sorted(_TRANSFORMS))
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("tvb", marks=pytest.mark.backend_tvb),
        pytest.param("jax", marks=pytest.mark.backend_jax),
        pytest.param("tvboptim", marks=pytest.mark.backend_tvboptim),
    ],
)
def test_the_sigmoidal_coupling_matches_tvb(tmp_path, backend, variant):
    if backend == "tvboptim":
        pytest.importorskip("tvboptim")
    exp = _experiment(tmp_path, variant)
    times, reference, states = _tvb_reference(WEIGHTS[variant], float(exp.integration.duration))

    trace = exp.run(backend).integration
    if "mode" in trace.dims:
        trace = trace.isel(mode=0)
    trace = trace.sel(variable=states).transpose("time", "variable", "node")

    np.testing.assert_allclose(trace["time"].values, times)
    np.testing.assert_allclose(trace.values, reference, rtol=1e-8, atol=1e-8)
