"""A swept cell is the experiment at the cell's values: it runs the declared settle itself, at those values, before its measured window.

``integration.transient_time`` is settling the experiment declares, so it belongs to every run of the experiment, and a cell of an independent sweep is such a run. A cell that opens its measured window on the base run's settled state instead records the relaxation from the base parameters to its own inside the window it reports, with nothing in the result to say so: for dx/dt = -(x - a) swept over a = 1 and a = 5 after a 40-unit settle, the a = 5 cell opened at 1.38 and reached 4.46 where it should have sat at 5.

The reference for every case here is the experiment run alone with the parameter set to the cell's value. That is the definition of a cell, so the comparison holds for whatever the experiment contains: delay lines, noise on either draw grain, a swept initial condition. The settle of a time-axis stochastic parameter is pinned separately, because it was held at its first sample in the base run as well.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

pytest.importorskip("tvboptim")

import jax

from tvbo import SimulationExperiment

RELAX = {
    "id": 1,
    "label": "relaxation to a swept fixed point",
    "dynamics": {
        "name": "Relax",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"a": {"value": 1.0}},
        "state_variables": {"x": {"equation": {"rhs": "-(x - a)"}, "initial_value": 0.0}},
    },
    "integration": {"method": "RungeKutta4thOrder", "step_size": 0.1, "duration": 2.0, "transient_time": 40.0},
    "explorations": {
        "sweep": {"name": "sweep", "mode": "product", "space": [{"parameter": "Relax.a", "explored_values": [1.0, 5.0]}]}
    },
}
"""dx/dt = -(x - a): the fixed point is a, reached to 4e-18 after the 40-unit settle, so a settled cell records x = a from its first sample."""

DELAYED = {
    "id": 2,
    "label": "two delayed phase oscillators",
    "dynamics": {
        "name": "Kuramoto",
        "parameters": {"omega": {"value": 0.0628}},
        "coupling_inputs": {"c": {"name": "c"}},
        "state_variables": {
            "theta": {
                "equation": {"lhs": "Derivative(theta, t)", "rhs": "omega + c"},
                "initial_value": 0.1,
                "variable_of_interest": True,
                "coupling_variable": True,
            }
        },
        "output": ["theta"],
    },
    "network": {
        "number_of_nodes": 2,
        "nodes": [{"id": 0, "label": "r0"}, {"id": 1, "label": "r1"}],
        "edges": [
            {"source": 0, "target": 1, "parameters": {"weight": {"value": 0.5}, "delay": {"value": 5.0}}},
            {"source": 1, "target": 0, "parameters": {"weight": {"value": 0.5}, "delay": {"value": 3.0}}},
        ],
        "coupling": {
            "KuramotoCoupling": {
                "delayed": True,
                "parameters": {"a": {"value": 0.5}, "N": {"value": 1.0}},
                "pre_expression": {"rhs": "sin(theta_j - theta_i)"},
                "post_expression": {"rhs": "a * gx / N"},
                "incoming_states": ["theta"],
                "local_states": ["theta"],
            }
        },
    },
    "integration": {"method": "RungeKutta4thOrder", "step_size": 1.0, "duration": 20.0, "transient_time": 30.0},
    "explorations": {
        "sweep": {
            "name": "sweep",
            "mode": "product",
            "space": [{"parameter": "Kuramoto.omega", "explored_values": [0.0628, 0.2]}],
        }
    },
}
"""Two oscillators coupled through unequal delays, so each cell's measured window reads delay lines that only that cell's own settle can have filled."""


def _noisy_relax(noise_draw):
    spec = copy.deepcopy(RELAX)
    spec["dynamics"]["state_variables"]["x"]["noise"] = {
        "additive": True,
        "gaussian": True,
        "parameters": {"sigma": {"value": 0.3}},
    }
    spec["integration"].update({"method": "Heun", "noise_draw": noise_draw, "block_size": 10})
    spec["execution"] = {"random_seed": 7}
    return spec


def _sweep_rows(spec, name="sweep"):
    rows = np.asarray(SimulationExperiment(**copy.deepcopy(spec)).run("tvboptim").explorations[name].results)
    return rows.reshape(rows.shape[0], -1)


def _integration(spec):
    return SimulationExperiment(**copy.deepcopy(spec)).run("tvboptim").integration


def _values(view):
    return np.asarray(getattr(view, "data", view))


def _measured_window(spec):
    return _values(_integration(spec).data).reshape(-1)


def _alone(spec, parameter, value):
    """The experiment without its sweep, with one dynamics parameter set to a cell's value."""
    alone = copy.deepcopy(spec)
    alone["dynamics"]["parameters"][parameter]["value"] = value
    del alone["explorations"]
    return alone


def test_swept_cell_settles_at_its_own_value():
    """Each cell of an independent sweep records the window after the declared transient at ITS value, not the base run's."""
    rows = _sweep_rows(RELAX)
    assert rows.shape[0] == 2
    for a, row in zip([1.0, 5.0], rows, strict=True):
        assert np.abs(row - a).max() < 1e-6, (
            f"cell a={a}: the measured window opens at {row[0]:.4f} and spans {row.min():.4f}..{row.max():.4f}; "
            f"after a 40-unit settle at a={a} it must sit at {a}"
        )


def test_swept_cell_equals_the_experiment_run_alone():
    """A swept cell is the same computation as the plain experiment with the parameter set to the cell's value."""
    np.testing.assert_allclose(_sweep_rows(RELAX)[1], _measured_window(_alone(RELAX, "a", 5.0)), atol=1e-9)


def test_swept_cell_of_a_delayed_network_continues_its_own_delay_lines():
    """The measured window of a cell reads the delay history its own settle wrote, on every edge."""
    rows = _sweep_rows(DELAYED)
    reference = _measured_window(_alone(DELAYED, "omega", 0.2))
    assert np.abs(rows[1] - rows[0]).max() > 1.0, "the two cells must differ for the comparison to mean anything"
    np.testing.assert_allclose(rows[1], reference, atol=1e-9)


@pytest.mark.parametrize("noise_draw", ["fused", "blocked"])
def test_noisy_swept_cell_equals_the_experiment_run_alone(noise_draw):
    """Under noise a cell draws what the experiment alone draws, in the settle and in the window, on either grain."""
    spec = _noisy_relax(noise_draw)
    rows = _sweep_rows(spec)
    reference = _measured_window(_alone(spec, "a", 5.0))
    assert reference.std() > 0.01, "the noise must reach the measured window for the comparison to mean anything"
    np.testing.assert_allclose(rows[1], reference, atol=1e-9)


def test_cell_swept_over_its_initial_condition_settles_from_it():
    """A swept initial condition is where the cell's settle opens: of two wells, each cell is found in the one its own start leads to.

    dx/dt = a (x - x^3) has stable points at -1 and +1 and the experiment declares its start at +0.5. A cell started at -0.5 settles to -1 only if its settle opens from -0.5; opened from the declared start it would sit at +1, and with no settle it would still be on its way.
    """
    spec = copy.deepcopy(RELAX)
    spec["dynamics"]["state_variables"]["x"] = {"equation": {"rhs": "a*(x - x**3)"}, "initial_value": 0.5}
    spec["explorations"]["sweep"]["space"] = [{"parameter": "initial_conditions.x", "explored_values": [-0.5, 0.5]}]
    rows = _sweep_rows(spec)
    for start, well, row in zip([-0.5, 0.5], [-1.0, 1.0], rows, strict=True):
        assert np.abs(row - well).max() < 1e-6, (
            f"cell started at {start}: the measured window spans {row.min():.4f}..{row.max():.4f} where it must sit at {well}"
        )


def test_the_base_run_is_the_experiment_run_alone():
    """Declaring a sweep changes nothing about the base run beside it."""
    alone = copy.deepcopy(DELAYED)
    del alone["explorations"]
    np.testing.assert_array_equal(_measured_window(DELAYED), _measured_window(alone))


def test_the_settle_stays_off_the_gradient_tape():
    """A gradient through a cell passes through its measured window only: the settle it opens from is a constant.

    For dx/dt = -(x - a) the window ends at a + (x0 - a) exp(-2), where x0 is the settled state. With the settle on the tape x0 = a differentiates to 1 and so does the endpoint; off it the endpoint differentiates to 1 - exp(-2).
    """
    generated = SimulationExperiment(**copy.deepcopy(RELAX)).execute("tvboptim")
    network = generated.create_network(np.zeros((1, 1)))
    sim = generated.run_simulation(network, run_main=False)
    own = generated._own_settle(network, sim.result_transient)

    def endpoint(a):
        state = sim.state.copy()
        state.dynamics.a = a
        state.initial_state.dynamics = sim.opening
        settled, _ = own(state)
        return sim.model_fn(settled).ys[-1].sum()

    assert float(endpoint(5.0)) == pytest.approx(5.0, abs=1e-6)
    assert float(jax.grad(endpoint)(5.0)) == pytest.approx(1.0 - math.exp(-2.0), abs=1e-6)


@pytest.mark.parametrize("missing", ["settle", "opening"])
def test_a_sweep_on_a_network_names_the_settle_and_opening_it_needs(missing):
    """Called on a network without the base run's settle, or without the state that settle opened from, a sweep with a declared transient says which two it needs."""
    generated = SimulationExperiment(**copy.deepcopy(RELAX)).execute("tvboptim")
    network = generated.create_network(np.zeros((1, 1)))
    sim = generated.run_simulation(network, run_main=False)
    given = {"settle": sim.result_transient, "opening": sim.opening}
    del given[missing]
    with pytest.raises(TypeError, match="needs the base run's `settle` and the `opening` state"):
        generated.sweep(sim.state, sim.model_fn, network=network, **given)


NOISY = copy.deepcopy(RELAX)
NOISY["dynamics"]["parameters"]["a"] = {
    "value": 1.0,
    "distribution": {"name": "Uniform", "domain": {"lo": "0", "hi": "2"}, "axis": "time"},
}
NOISY["dynamics"]["state_variables"]["x"]["equation"]["rhs"] = "-5*(x - a)"
NOISY["integration"]["transient_time"] = 20.0
del NOISY["explorations"]

DRIVEN = copy.deepcopy(NOISY)
DRIVEN["dynamics"]["state_variables"]["x"]["equation"]["rhs"] = "a"
DRIVEN["integration"].update({"method": "euler", "transient_time": 40.0})
"""dx/dt = a(t) under Euler: each step adds dt times the sample it read, so the increments of the trajectory are the samples, in the order they were read."""


def test_time_axis_stochastic_parameter_drives_the_settle():
    """The settle of a parameter drawn along time is driven by the draw, not held at its first sample."""
    integ = _integration(NOISY)
    settle, measured = _values(integ.transient).reshape(-1), _values(integ.data).reshape(-1)
    assert settle[-100:].std() > 0.5 * measured.std(), (
        f"settle tail std {settle[-100:].std():.2e} against measured-window std {measured.std():.2e}: the settle saw a constant parameter"
    )


def test_settle_and_window_read_one_draw_sample_by_sample():
    """Every step of the settle and of the window after it reads its own sample of the one draw: none twice, none skipped into the other's."""
    integ = _integration(DRIVEN)
    trajectory = np.concatenate([[0.0], _values(integ.transient).reshape(-1), _values(integ.data).reshape(-1)])
    samples = np.diff(trajectory) / 0.1
    assert samples.size == 420
    assert samples.min() >= -1e-9 and samples.max() <= 2.0 + 1e-9
    assert np.unique(np.round(samples, 9)).size == samples.size, (
        f"{samples.size - np.unique(np.round(samples, 9)).size} of {samples.size} steps read a sample another step also read"
    )


STIMULATED = {
    "id": 6,
    "label": "a node integrating an iid stimulus",
    "dynamics": {
        "name": "Integrator",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"a": {"value": 0.0}},
        "state_variables": {"x": {"equation": {"rhs": "a"}, "initial_value": 0.0}},
    },
    "integration": {"method": "euler", "step_size": 0.1, "duration": 2.0, "transient_time": 40.0},
    "events": {
        "u": {
            "event_type": "stimulation",
            "equation": {"rhs": "u"},
            "parameters": {
                "u": {
                    "name": "u",
                    "value": 0.0,
                    "distribution": {"name": "Uniform", "axis": "time", "domain": {"lo": 0.0, "hi": 2.0}, "seed": 7},
                }
            },
            "target_variable": "x",
        }
    },
}
"""dx/dt = u(t) under Euler, u a stimulus drawn once per step: the increments of the trajectory are the samples the stimulus handed over, in the order it handed them."""


@pytest.mark.parametrize(("settle", "dt", "duration"), [(40.0, 0.1, 2.0), (0.5, 0.002, 0.5)])
def test_an_iid_stimulus_hands_each_step_its_own_sample(settle, dt, duration):
    """Step k of the settle and the window after it reads sample k of the one sequence, the sequence a study rebuilds from the declared seed and bounds to form its targets."""
    spec = copy.deepcopy(STIMULATED)
    spec["integration"].update({"step_size": dt, "duration": duration, "transient_time": settle})
    integ = _integration(spec)
    trajectory = np.concatenate([[0.0], _values(integ.transient).reshape(-1), _values(integ.data).reshape(-1)])
    samples = np.diff(trajectory) / dt
    steps = round((settle + duration) / dt)
    assert samples.size == steps
    drawn = np.asarray(jax.random.uniform(jax.random.key(7), (steps + 2,), minval=0.0, maxval=2.0))[:steps]
    off = np.flatnonzero(np.abs(samples - drawn) > 5e-4)
    assert off.size == 0, f"{off.size} of {steps} steps read another step's sample, the first at step {off[:1]}"
