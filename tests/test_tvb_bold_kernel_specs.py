"""Each curated BOLD spec that names a TVB `hrf_kernel` declares that kernel's own equation and parameter names, so the tvb backend builds the HRF TVB builds.

The tvb backend hands the pipeline's equation parameters to `tvb.datatypes.equations.<kernel>`; a spec that names them differently (`mean`/`var` for `Gamma`'s `tau`/`n`) makes TVB's `evaluate` raise a `KeyError`. The reference is TVB's kernel with its defaults, on the Bold monitor's stock grid.
"""

import ast

import numpy as np
import pytest

equations = pytest.importorskip("tvb.datatypes.equations")

from tvbo import Coupling, Dynamics, Network, Observation, SimulationExperiment

pytestmark = pytest.mark.backend_tvb

STOCK_TIME = np.arange(0.0, 20.0, 20.0 / 5000)


def _emitted_kernel(spec):
    """The HRF kernel object the tvb backend's generated monitor for *spec* constructs, built by executing its own statements."""
    network = Network.from_matrix(
        weights=np.array([[0.0, 0.2], [0.1, 0.0]]), lengths=np.zeros((2, 2)), labels=["left", "right"]
    )
    network.coupling["Linear"] = Coupling.from_db("Linear")
    observation = Observation.from_db(spec)
    observation.source = ["S"]
    experiment = SimulationExperiment(
        dynamics=Dynamics.from_db("ReducedWongWang"),
        network=network,
        integration={"method": "Heun", "duration": 1000.0, "step_size": 0.1},
        observations=[observation],
    )
    tree = ast.parse(experiment.render_code("tvb"))
    compute_hrf = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_compute_hrf")
    statements = [ast.unparse(st) for st in compute_hrf.body if ast.unparse(st).startswith(("hrf_kernel =", "hrf_kernel."))]
    scope = {"equations": equations}
    exec("\n".join(statements), scope)
    return scope["hrf_kernel"]


@pytest.mark.parametrize(
    ("spec", "kernel"),
    [
        ("BOLD_TVB", "FirstOrderVolterra"),
        ("BOLD_Gamma", "Gamma"),
        ("BOLD_DoubleExponential", "DoubleExponential"),
        ("BOLD_MixtureOfGammas", "MixtureOfGammas"),
    ],
)
def test_the_spec_builds_tvbs_own_kernel(spec, kernel):
    got = _emitted_kernel(spec).evaluate(STOCK_TIME)
    expected = getattr(equations, kernel)().evaluate(STOCK_TIME)
    assert np.array_equal(got, expected)
