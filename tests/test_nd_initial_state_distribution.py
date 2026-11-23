"""A declared initial-state distribution draws on NetworkDynamics.jl by the definition every backend that draws an initial state shares.

The rendered Julia draws each node's initial value of a state variable that declares a distribution from the family and parameters `initial_state_distribution` gives: a normal centred between the bounds with a quarter of their width as spread, ``loc .+ scale .* randn(rng)``, or a uniform between them, ``lo .+ (hi - lo) .* rand(rng)``, the bounds the distribution's own `domain`, else the state variable's, else (-1, 1). What that definition refuses is refused when the Julia is rendered: another family, a declared `mean` or `std`, and bounds that are not finite or not in order. A dynamics parameter drawn over time (`axis: time`), which the render would draw once per node, is refused too. Only the render is read; no Julia runs.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

from tvbo import SimulationExperiment

from .test_initial_state_distribution import ONE_VARIABLE, REFUSED, SEED, _experiment

HETEROGENEOUS = (
    Path(__file__).resolve().parents[1] / "docs/Interoperability/NetworkDynamics.jl/yaml/heterogeneous_kuramoto.yaml"
)
NUMBER = r"(-?[\d.]+(?:e[-+]?\d+)?)"


def _draw(code, variable, row="node"):
    """The family and parameters of the Julia that draws *variable* on the row *row* matches: ``("normal", loc, scale)`` or ``("uniform", lo, hi)``."""
    line = re.search(rf"s\.v\[{row}, :{variable}\] = (.+)", code)
    assert line, f"the render draws no initial {variable}"
    uniform = re.fullmatch(rf"{NUMBER} \.\+ \({NUMBER} - {NUMBER}\) \.\* rand\(rng\)", line[1].strip())
    if uniform:
        return "uniform", float(uniform[1]), float(uniform[2])
    normal = re.fullmatch(rf"{NUMBER} \.\+ {NUMBER} \.\* randn\(rng\)", line[1].strip())
    assert normal, f"{variable} is drawn by {line[1].strip()!r}"
    return "normal", float(normal[1]), float(normal[2])


@pytest.mark.parametrize(
    "distribution, domain, expected",
    [
        ({"name": "Normal", "domain": {"lo": 1.0, "hi": 2.0}}, None, ("normal", 1.5, 0.25)),
        ({"name": "Gaussian", "domain": {"lo": -2.0, "hi": 2.0}}, None, ("normal", 0.0, 1.0)),
        ({"name": "Uniform", "domain": {"lo": 0.5, "hi": 3.0}}, None, ("uniform", 0.5, 3.0)),
        ({"name": "Uniform"}, {"lo": -2.0, "hi": 4.0}, ("uniform", -2.0, 4.0)),
        ({"name": "Uniform"}, None, ("uniform", -1.0, 1.0)),
    ],
    ids=["normal-own-domain", "gaussian-own-domain", "uniform-own-domain", "state-variable-domain", "default-bounds"],
)
def test_the_julia_draws_the_shared_family_and_parameters(distribution, domain, expected):
    """Each node's ``x`` is drawn from the module's one stream, seeded with the run's seed, by the family and parameters `initial_state_distribution` gives."""
    code = _experiment(distribution, domain).render_code("networkdynamics")
    assert f"rng = MersenneTwister({SEED})" in code
    assert _draw(code, "x") == expected


def test_a_node_of_another_dynamics_draws_by_the_same_definition():
    """The docs' heterogeneous Kuramoto network, its inertia node's ``omega`` declaring Normal on [4, 6] and no initial value of its own, draws it as normal(5, 0.5)."""
    spec = yaml.safe_load(HETEROGENEOUS.read_text())
    omega = spec["network"]["dynamics"]["KuramotoInertia"]["state_variables"]["omega"]
    omega["distribution"] = {"name": "Normal", "domain": {"lo": 4.0, "hi": 6.0}}
    (inertia,) = [node for node in spec["network"]["nodes"] if node.get("dynamics") == "KuramotoInertia"]
    del inertia["state"]["omega"]
    code = SimulationExperiment(**spec).render_code("networkdynamics")
    assert _draw(code, "omega", row=r"\d+") == ("normal", 5.0, 0.5)


@pytest.mark.parametrize(
    "distribution, domain, message", REFUSED, ids=["unbounded", "inverted", "other-family", "declared-moments"]
)
def test_the_julia_render_refuses_what_the_shared_definition_refuses(distribution, domain, message):
    """Bounds from which a draw would be infinite (a half-open state variable under a distribution without a domain) or empty (inverted), a family other than uniform or normal, and a `mean`/`std` the bounds stand in for are refused by name."""
    with pytest.raises(ValueError, match=message):
        _experiment(distribution, domain).render_code("networkdynamics")


def test_the_julia_render_refuses_a_parameter_drawn_over_time():
    """A dynamics parameter whose distribution declares `axis: time` is a stochastic input, which the render would draw once per node and never per step; it is refused by name."""
    spec = copy.deepcopy(ONE_VARIABLE)
    over_time = {"name": "Gaussian", "axis": "time", "domain": {"lo": 0.0, "hi": 2.0}}
    spec["dynamics"]["parameters"]["a"]["distribution"] = over_time
    message = r"never per step, so parameter 'a' of dynamics 'Drawn', declared with `axis: time`"
    with pytest.raises(NotImplementedError, match=message):
        SimulationExperiment(**spec).render_code("networkdynamics")
