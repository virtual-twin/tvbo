"""DifferentialEquations.jl, ModelingToolkit.jl, the python backend and gillespie open on a declared initial-state distribution.

The two Julia renders draw each declared state variable with the Julia NetworkDynamics.jl draws it with (`initial_state_draws`), from one `MersenneTwister` seeded with the distributions' own seed, else the run's; python, on one node or a network, and gillespie open on `get_initial_values` on the run's seed, the draw tvb and jax open on. The Julia renders are read, not run; the python and gillespie runs integrate a state that does not move, so every recorded sample is the opening.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tvbo import SimulationExperiment

from .test_initial_state_distribution import N_NODES, ONE_VARIABLE, SEED

UNIFORM = {"name": "Uniform", "domain": {"lo": 0.5, "hi": 3.0}}
NORMAL = {"name": "Normal", "domain": {"lo": 1.0, "hi": 2.0}}

GILLESPIE = {
    "label": "a relaxing activity beside a held variable",
    "dynamics": {
        "name": "Held",
        "parameters": {"tau": {"value": 1.0}, "c": {"value": 0.5}},
        "state_variables": {
            "X": {"equation": {"rhs": "(-X + c)/tau"}, "initial_value": 0.5},
            "y": {"equation": {"rhs": "0*y"}, "initial_value": 0.0, "distribution": UNIFORM},
        },
        "output": ["X", "y"],
    },
    "integration": {"method": "Euler", "step_size": 0.1, "duration": 1.0},
    "execution": {"system_size": 100.0, "random_seed": SEED},
}


def _held(distribution, network=True):
    """G240's one-variable recipe with ``x`` held still (``a = 0``) and declaring *distribution*, on its five nodes or, without *network*, as one system."""
    spec = copy.deepcopy(ONE_VARIABLE)
    spec["dynamics"]["parameters"]["a"]["value"] = 0.0
    spec["dynamics"]["state_variables"]["x"]["distribution"] = distribution
    if not network:
        del spec["network"]
    return SimulationExperiment(**spec)


@pytest.mark.parametrize("backend, prefix", [("julia", ""), ("modelingtoolkit", "sys.x => ")], ids=["diffeq", "mtk"])
@pytest.mark.parametrize(
    "distribution, draw",
    [(UNIFORM, "0.5 .+ (3.0 - 0.5) .* rand(rng)"), (NORMAL, "1.5 .+ 0.25 .* randn(rng)")],
    ids=["uniform", "normal"],
)
def test_the_julia_render_opens_on_the_shared_draw(backend, prefix, distribution, draw):
    """The render seeds one stream with the run's seed and opens ``x`` on the line NetworkDynamics.jl draws it with."""
    code = _held(distribution, network=False).render_code(backend)
    assert f"rng = MersenneTwister({SEED})" in code
    assert f"{prefix}{draw}" in code


def test_the_julia_render_refuses_two_seeds_on_one_stream():
    """Two declared initial states under two seeds cannot be drawn from the module's one stream; the render says so by name."""
    spec = copy.deepcopy(ONE_VARIABLE)
    del spec["network"]
    variables = spec["dynamics"]["state_variables"]
    variables["x"]["distribution"] = {**UNIFORM, "seed": 11}
    variables["z"] = {"equation": {"rhs": "-z"}, "initial_value": 0.0, "distribution": {**UNIFORM, "seed": 12}}
    message = r"seeded once, and state variable 'x' resolves to seed 11, state variable 'z' resolves to seed 12"
    with pytest.raises(NotImplementedError, match=message):
        SimulationExperiment(**spec).render_code("julia")


@pytest.mark.parametrize("network, n_nodes", [(False, 1), (True, N_NODES)], ids=["one-node", "network"])
def test_the_python_backend_opens_on_the_draw_tvb_and_jax_open_on(network, n_nodes):
    """Every recorded sample of the held ``x`` is the uniform draw `get_initial_values` makes on the run's seed, the generator G237 keys."""
    data = _held(UNIFORM, network).run("python").integration.data
    reference = np.random.default_rng((SEED, 0)).uniform(0.5, 3.0, n_nodes)
    held = data.sel(variable="x").values.reshape(data.sizes["time"], -1)
    np.testing.assert_array_equal(held, np.broadcast_to(reference, held.shape))


def test_gillespie_opens_on_the_draw_tvb_and_jax_open_on():
    """The held slow variable ``y`` keeps the uniform draw `get_initial_values` makes on the run's seed in every recorded sample."""
    data = SimulationExperiment(**copy.deepcopy(GILLESPIE)).run("gillespie").integration.data
    reference = np.random.default_rng((SEED, 0)).uniform(0.5, 3.0, 1)[0]
    np.testing.assert_array_equal(data.sel(variable="y").values.ravel(), reference)
