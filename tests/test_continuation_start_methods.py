"""A continuation starts by ``time_integration``, ``given`` or ``newton``, the same on every continuation backend.

BifurcationKit and PyRates settled every continuation by time integration whatever it declared, ``given`` started only AUTO-07p at the declared state, and ``newton`` ran nowhere: each backend integrated instead. AUTO-07p read an undeclared initial value as 0 where the others read 0.1, PyRates settled for 10000 time units where the schema's default is 2000, and a method that seeds a simulation ran as a time integration.

The model is ``x' = a - x``, ``y' = -2 y``, whose equilibrium is ``(a, 0)``, so each start is known in closed form.
"""

from __future__ import annotations

import os

import pytest

from tvbo import SimulationExperiment
from tvbo.adapters.base import ContinuationAdapter
from tvbo.adapters.bifurcationkit import BifurcationKitAdapter
from tvbo.adapters.numcont import NumContAdapter, _warmup_to_steady_state
from tvbo.adapters.pyrates_bifurcation import PyRatesBifurcationAdapter

BACKENDS = (BifurcationKitAdapter, NumContAdapter, PyRatesBifurcationAdapter)
A = 1.0


def _model(x0=A, rhs="a - x"):
    return {
        "name": "Shift",
        "parameters": {"a": {"value": A, "domain": {"lo": -2.0, "hi": 2.0}}},
        "state_variables": {
            "x": {"equation": {"rhs": rhs}, **({} if x0 is None else {"initial_value": x0})},
            "y": {"equation": {"rhs": "-2*y"}, "initial_value": 0.0},
        },
    }


def _experiment(method=None, x0=A, rhs="a - x", extra_states=0):
    continuation = {"name": "eq", "free_parameters": [{"name": "a"}], "ds": 0.05, "max_steps": 200, "bothside": True}
    if method is not None:
        continuation["initial_state"] = {"method": method, "duration": 50.0}
    model = _model(x0, rhs)
    for k in range(extra_states):
        model["state_variables"][f"z{k}"] = {"equation": {"rhs": f"-z{k}"}, "initial_value": 0.0}
    return SimulationExperiment(dynamics=model, continuations={"eq": continuation})


def _started(backend, experiment):
    """The dynamics *backend* hands `run_one` for *experiment*'s continuation."""
    adapter = backend(experiment)
    adapter.run_one = lambda model, cont, name, **kwargs: model
    return adapter.run()


def _initial(model):
    return {name: state.initial_value for name, state in model.state_variables.items()}


def test_given_starts_every_backend_at_the_declared_state_without_a_settle():
    experiment = _experiment("given")
    cont = experiment.continuations["eq"]
    assert list(_warmup_to_steady_state(experiment.dynamics, cont)) == [A, 0.0]
    bifurcationkit = BifurcationKitAdapter(experiment).render_code()
    assert "x0_eq = x0\n" in bifurcationkit and "_find_steady_state" not in bifurcationkit
    pyrates = PyRatesBifurcationAdapter(experiment).render_code()
    assert 'starting_point="EP1"' in pyrates and "NMX=1\n" in pyrates and "UZR" not in pyrates


def test_time_integration_settles_every_backend_for_the_declared_duration():
    experiment = _experiment("time_integration", x0=0.0)
    assert _warmup_to_steady_state(experiment.dynamics, experiment.continuations["eq"])[0] == pytest.approx(A, abs=1e-8)
    assert "function _find_steady_state(f!, x0, p; T=50.0)" in BifurcationKitAdapter(experiment).render_code()
    pyrates = PyRatesBifurcationAdapter(experiment).render_code()
    assert "UZR={14: 50.0}" in pyrates and 'starting_point="UZ1"' in pyrates


def test_pyrates_refuses_a_settle_solver_it_cannot_integrate_by():
    """PyRates settles by AUTO-07p's initial-value run, so a declared solver would go unused without a word."""
    continuation = {"name": "eq", "free_parameters": [{"name": "a"}], "initial_state": {"solver": {"method": "Tsit5"}}}
    experiment = SimulationExperiment(dynamics=_model(), continuations={"eq": continuation})
    with pytest.raises(ValueError, match="solver 'Tsit5'"):
        PyRatesBifurcationAdapter(experiment).render_code()


def test_pyrates_settles_for_the_schemas_duration_where_none_is_declared():
    assert "UZR={14: 2000.0}" in PyRatesBifurcationAdapter(_experiment()).render_code()


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda cls: cls.__name__)
def test_newton_starts_every_backend_at_the_equilibrium_near_the_declared_state(backend):
    experiment = _experiment("newton", x0=1.5)
    assert _initial(_started(backend, experiment)) == pytest.approx({"x": A, "y": 0.0}, abs=1e-12)
    assert _initial(experiment.dynamics)["x"] == 1.5


def test_newton_names_a_model_it_finds_no_equilibrium_of():
    experiment = _experiment("newton", rhs="1 + x**2")
    with pytest.raises(RuntimeError, match="'newton' found no equilibrium of 'Shift'"):
        NumContAdapter(experiment).render_code()


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda cls: cls.__name__)
def test_given_refuses_a_state_that_is_not_an_equilibrium(backend):
    """AUTO-07p took such a state as the first point of its branch, and BifurcationKit corrected it by Newton's method first."""
    adapter = backend(_experiment("given", x0=1.5))
    for call in (adapter.render_code, adapter.run):
        with pytest.raises(
            ValueError, match="'given' from a state of 'Shift' that is not an equilibrium: its right-hand side reaches 0.5"
        ):
            call()


@pytest.mark.parametrize("method", ["from_working_point", "from_experiment"])
@pytest.mark.parametrize("backend", BACKENDS, ids=lambda cls: cls.__name__)
def test_a_method_that_seeds_a_simulation_is_refused(backend, method):
    with pytest.raises(ValueError, match=f"{method}', which seeds a simulation"):
        backend(_experiment(method)).render_code()


def test_every_backend_reads_an_undeclared_initial_value_as_the_same_state():
    """``x' = 0.1 - x`` rests at the 0.1 an undeclared initial value reads as, which AUTO-07p read as 0."""
    experiment = _experiment("given", x0=None, rhs="0.1 - x")
    assert list(_warmup_to_steady_state(experiment.dynamics, experiment.continuations["eq"])) == [0.1, 0.0]
    assert list(ContinuationAdapter.declared_state(experiment.dynamics)) == [0.1, 0.0]


def _branch(backend, experiment):
    return experiment.run(backend).continuations["eq"].df


BRANCH_BACKENDS = [
    pytest.param("auto-07p", marks=pytest.mark.slow),
    pytest.param("pyrates-bifurcation", marks=[pytest.mark.slow, pytest.mark.backend_pyrates]),
    pytest.param("bifurcationkit", marks=[pytest.mark.slow, pytest.mark.julia, pytest.mark.backend_julia]),
]


@pytest.mark.parametrize("method", ["given", "newton", "time_integration"])
@pytest.mark.parametrize("backend", BRANCH_BACKENDS)
def test_every_start_continues_the_equilibrium_branch(backend, method, monkeypatch, tmp_path):
    """From the equilibrium (given), from beside it (newton) or from the origin (time_integration), every backend continues ``x = a`` across the parameter's domain."""
    if backend != "bifurcationkit" and not os.environ.get("AUTO_DIR"):
        pytest.skip("AUTO_DIR is not set")
    pytest.importorskip("juliacall" if backend == "bifurcationkit" else "auto" if backend == "auto-07p" else "pycobi")
    monkeypatch.chdir(tmp_path)
    x0 = {"given": A, "newton": 1.5, "time_integration": 0.0}[method]
    df = _branch(backend, _experiment(method, x0=x0))
    assert len(df) > 10
    assert (df["x"] - df["param"]).abs().max() < 1e-8
    assert df["param"].min() < -1.9 and df["param"].max() > 1.9


@pytest.mark.slow
@pytest.mark.backend_pyrates
def test_a_numcont_continuation_starts_from_none_of_the_previous_runs(monkeypatch, tmp_path):
    """AUTO-07p reuses the constants and solution of the process's last run where a run names no starting solution, so a three-state model continued on auto-07p after PyRates continued a two-state one raised ``IndexError`` in AUTO's solution parser."""
    if not os.environ.get("AUTO_DIR"):
        pytest.skip("AUTO_DIR is not set")
    pytest.importorskip("pycobi")
    monkeypatch.chdir(tmp_path)
    _branch("pyrates-bifurcation", _experiment("given"))
    df = _branch("auto-07p", _experiment("given", extra_states=1))
    assert (df["x"] - df["param"]).abs().max() < 1e-8
