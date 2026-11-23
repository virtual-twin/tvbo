"""PyRates, NeuroML/LEMS and the CUDA kernel open on the draw tvb and jax open on, written into what they render, and refuse a network whose nodes the render cannot draw apart.

Each of these renders writes one number per state variable, which was the declared `initial_value` and is now the draw `get_initial_values` makes on the run's seed. G240's one-variable recipe declares ``x`` uniform on [0.5, 3.0] with an initial value of 0, so reading the render tells the drawn number from the declared one. The same recipe on five nodes, which tvb and jax draw apart, is refused by each, naming the state variable. A standard NeuroML cell type opens from its own start and parameters and refuses a declared distribution, but `fitzHughNagumo1969Cell`, whose ``V0`` carries the draw; a PyRates continuation still starts from the initial value.
"""

from __future__ import annotations

import copy
import re

import pytest

from tvbo import SimulationExperiment
from tvbo.codegen.pyrates import to_pyrates_yaml_string

from .test_initial_state_distribution import N_NODES, ONE_VARIABLE

UNIFORM = {"name": "Uniform", "domain": {"lo": 0.5, "hi": 3.0}}
RENDERS = {
    "pyrates": (lambda experiment: experiment.to_yaml(format="pyrates"), "x: variable({})"),
    "lems": (lambda experiment: experiment.render_code("lems"), 'x_0="{}"'),
    "cuda": (lambda experiment: experiment.render_code("cuda"), "float x = {}f;"),
}
BACKEND = {"pyrates": "pyrates", "lems": "neuroml", "cuda": "rateml-cuda"}


def _recipe(distribution=UNIFORM, network=False):
    """G240's one-variable recipe, ``x`` declaring *distribution*, on one node or on its five."""
    spec = copy.deepcopy(ONE_VARIABLE)
    if distribution:
        spec["dynamics"]["state_variables"]["x"]["distribution"] = distribution
    if not network:
        del spec["network"]
    return SimulationExperiment(**spec)


def _opening(experiment, name="x"):
    """*name* as tvb and jax open it on one node, the run's seed keying the draw."""
    model = experiment.dynamics
    values = model.get_initial_values(seed=experiment.execution.random_seed).ravel()
    return float(values[list(model.state_variables).index(name)])


@pytest.mark.parametrize("key", list(RENDERS))
def test_a_render_writes_the_host_draw(key):
    """The number the render writes for ``x`` is the draw tvb and jax open on, not the initial value 0."""
    render, written = RENDERS[key]
    experiment = _recipe()
    assert written.format(_opening(experiment)) in render(experiment)


def test_an_undeclared_distribution_keeps_the_initial_value():
    """Without a distribution the opening is the declared initial value, as before."""
    assert 'x_0="0.0"' in _recipe(distribution=None).render_code("lems")


@pytest.mark.parametrize("key", list(RENDERS))
def test_a_network_the_render_cannot_draw_apart_is_refused(key):
    """Five nodes share the one number the render writes, where tvb and jax draw each apart, so the render refuses, naming ``x``."""
    render, _ = RENDERS[key]
    expected = (
        f"the {BACKEND[key]} backend writes one opening per state variable, shared by all {N_NODES} instances of the model, "
        "so the initial-state distribution declared on 'x' cannot reach the run"
    )
    with pytest.raises(NotImplementedError, match=re.escape(expected)):
        render(_recipe(network=True))


def _standard_cell(iri, state_variables, parameters):
    """One node of a standard NeuroML cell type *iri*."""
    return SimulationExperiment(
        dynamics={"name": "Cell", "iri": iri, "parameters": parameters, "state_variables": state_variables},
        execution={"random_seed": 7},
        integration={"method": "Euler", "step_size": 0.1, "duration": 1.0},
    )


def test_a_standard_cell_refuses_a_declared_distribution():
    """An izhikevich2007Cell opens from the type's own start and parameters, which have no place for a draw."""
    cell = _standard_cell(
        "neuroml:izhikevich2007Cell",
        {"v": {"equation": {"rhs": "0"}, "initial_value": -60.0, "distribution": UNIFORM}},
        {"v0": {"value": -60.0}},
    )
    expected = "the neuroml backend opens a standard izhikevich2007Cell from the type's own start and parameters, so the initial-state distribution declared on 'v'"
    with pytest.raises(NotImplementedError, match=re.escape(expected)):
        cell.render_code("neuroml")


def test_fitzhugh_nagumo_1969_writes_the_draw_into_v0():
    """`fitzHughNagumo1969Cell` opens at its ``V0`` and ``W0``, which carry the run's draw."""
    cell = _standard_cell(
        "neuroml:fitzHughNagumo1969Cell",
        {
            "V": {"equation": {"rhs": "0"}, "initial_value": 0.0, "distribution": UNIFORM},
            "W": {"equation": {"rhs": "0"}, "initial_value": 0.0},
        },
        {name: {"value": value} for name, value in {"a": 0.7, "b": 0.8, "I": 0.5, "phi": 0.08}.items()},
    )
    assert f'V0="{_opening(cell, "V")}"' in cell.render_code("neuroml")


def test_a_pyrates_continuation_starts_from_the_initial_value():
    """The AUTO-07p continuation reads the PyRates operator's opening as its Newton start, which stays the declared initial value."""
    assert "x: variable(0.0)" in to_pyrates_yaml_string(_recipe().dynamics, fortran=True, seed=None)
