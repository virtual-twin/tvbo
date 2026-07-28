"""Gillespie SSA backend — finite-size stochastic realization of a mean-field rate model.

The backend (``tvbo/adapters/gillespie.py``) runs any relaxation rate model ``tau*X' = -X + F``
as a finite birth-death process: it derives the birth/death split (birth ``Omega*F/tau``, death
``n/tau``) and the between-event ODEs from the model's *own* equations — nothing model-specific.
These tests pin: (1) the generic decomposition is correct on a Tsodyks-Markram rate model;
(2) the mean-field limit — large system size ``Omega`` concentrates the activity on the
deterministic fixed point (which only holds if the split is right); (3) noise scales as
``1/sqrt(Omega)``; (4) the backend requires ``execution.system_size``.

The model is declared inline (no external recipe) so the test is self-contained.
"""
from __future__ import annotations

import numpy as np
import pytest

from tvbo.classes.experiment import SimulationExperiment

# A single-node Tsodyks-Markram rate model in a deep, non-bursting down-state (I0 = -2.0),
# so the deterministic mean field has one low stable fixed point to converge to.
TM_YAML = """
label: "Tsodyks-Markram rate model (gillespie backend test)"
dynamics:
  name: TMrate
  model_type: neural_mass
  parameters:
    J: {value: 3.07}
    alpha: {value: 1.5}
    E0: {value: -2.0}
    tau: {value: 0.013}
    tauD: {value: 0.2}
    tauF: {value: 1.5}
    U0: {value: 0.3}
  derived_variables:
    SS0: {equation: {rhs: "J*u*x*E + E0"}}
    SS1: {equation: {rhs: "alpha*log(1 + exp(SS0/alpha))"}}
  state_variables:
    E: {equation: {lhs: "Derivative(E, t)", rhs: "(-E + SS1)/tau"}, initial_value: 0.5, domain: {lo: 0.0, hi: 60.0}}
    x: {equation: {lhs: "Derivative(x, t)", rhs: "(1 - x)/tauD - u*x*E"}, initial_value: 0.98, domain: {lo: 0.0, hi: 1.0}}
    u: {equation: {lhs: "Derivative(u, t)", rhs: "(U0 - u)/tauF + U0*(1 - u)*E"}, initial_value: 0.37, domain: {lo: 0.0, hi: 1.0}}
  output: [E, x, u]
integration:
  method: Euler
  duration: 150.0
  step_size: 0.05
  unit: s
execution:
  backend: gillespie
  system_size: 30.0
  random_seed: 0
"""


def _experiment(tmp_path, system_size=30.0, seed=0):
    path = tmp_path / "tm_gillespie.yaml"
    path.write_text(TM_YAML)
    exp = SimulationExperiment.from_file(str(path))
    if system_size is None:
        exp.execution.system_size = None
    else:
        exp.execution.system_size = float(system_size)
    exp.execution.random_seed = int(seed)
    return exp


def _deterministic_downstate(J=3.07, alpha=1.5, E0=-2.0, tauD=0.2, tauF=1.5, U0=0.3):
    """Steady state of the deterministic mean field (Euler to convergence)."""
    E, x, u, dt = 0.5, 0.98, 0.37, 1e-3
    g = lambda w: alpha * np.log1p(np.exp(w / alpha))
    for _ in range(400_000):
        E += (-E + g(J * u * x * E + E0)) / 0.013 * dt
        x += ((1 - x) / tauD - u * x * E) * dt
        u += ((U0 - u) / tauF + U0 * (1 - u) * E) * dt
    return E


def _run_E(exp):
    da = exp.run(format="gillespie").integration.data
    return np.asarray(da.sel(variable="E").squeeze().values)


def test_generic_decomposition(tmp_path):
    """The adapter derives (tau, birth flux, slow RHS) from the equations, not from hardcoded TM."""
    from tvbo.adapters.gillespie import GillespieAdapter

    exp = _experiment(tmp_path)
    exp.configure()
    activity, sv_names, tau, birth_fn, slow_fns = GillespieAdapter(exp)._compile(exp.dynamics)
    assert activity == "E"                       # first output = activity variable
    assert set(slow_fns) == {"x", "u"}           # the rest integrate deterministically
    assert tau == pytest.approx(0.013, rel=1e-9)  # leak coefficient -> relaxation time
    # birth flux F/tau at the down-state equals the deterministic gain SS1/tau
    Estar = _deterministic_downstate()
    x0 = 1.0 / (1.0 + 0.2 * 0.37 * Estar)
    args = {"E": Estar, "x": x0, "u": 0.37}
    ss1 = 1.5 * np.log1p(np.exp((3.07 * 0.37 * x0 * Estar - 2.0) / 1.5))
    got = float(birth_fn(*[args[n] for n in sv_names]))
    assert got == pytest.approx(ss1 / 0.013, rel=1e-6)


def test_mean_field_limit(tmp_path):
    """Large Omega concentrates the activity on the deterministic fixed point."""
    E = _run_E(_experiment(tmp_path, system_size=60.0, seed=1))
    E = E[len(E) // 5:]                            # drop the transient
    assert E.mean() == pytest.approx(_deterministic_downstate(), rel=0.20)


def test_noise_scales_with_system_size(tmp_path):
    """Smaller Omega -> stronger finite-size fluctuations (noise ~ 1/sqrt(Omega))."""
    std_small = _run_E(_experiment(tmp_path, system_size=6.0, seed=2)).std()
    std_large = _run_E(_experiment(tmp_path, system_size=60.0, seed=2)).std()
    assert std_small > std_large


def test_requires_system_size(tmp_path):
    """The backend refuses to run without execution.system_size."""
    exp = _experiment(tmp_path, system_size=None)
    with pytest.raises(ValueError, match="system_size"):
        exp.run(format="gillespie")
