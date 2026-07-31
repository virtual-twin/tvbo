"""Regression tests for ``tvbo.analysis.linear_response`` and its codegen partials.

The linear-response path builds the network Jacobian ``A`` from a model's dfun metadata
and solves the Lyapunov equation ``A P + P A^T + Q = 0`` for the stationary covariance.
Two things about that are metadata-driven and were not, until these tests, pinned:

* ``Q`` must come from the noise each state variable DECLARES. A model whose noise enters
  two of six equations — synaptic gating plus a haemodynamic cascade that is *driven*,
  not forced — gets noise injected into its haemodynamics under a uniform ``Q``.
* the covariance that comes out must be of the DECLARED observable, which may be a derived
  variable at the end of an observation cascade (a BOLD signal), not only the first state
  block.

Both are checked against a NumPy/scipy reference, and the pre-existing uniform-``Q``,
first-state-block behaviour is pinned so it cannot drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvbo.analysis.linear_response import (
    linear_response_context,
    network_jacobian,
    noise_terms,
    observable_terms,
)

_SPEC = """
name: LRProbe
output: [x]
parameters:
  a: {value: 1.7}
  b: {value: 0.9}
  k: {value: 2.5}
  m: {value: -0.4}
coupling_inputs:
  c_glob: {}
derived_variables:
  drive: {equation: {rhs: "m * x"}}
  obs: {equation: {rhs: "k * y + drive"}}
state_variables:
  x:
    equation: {rhs: "-a * x + c_glob"}
    initial_value: 0.0
    coupling_variable: true
    noise:
      additive: true
      parameters: {sigma: {value: 0.02}}
  y:
    equation: {rhs: "b * (x - y)"}
    initial_value: 0.0
number_of_modes: 1
"""


def _model(spec=_SPEC):
    from tvbo.classes.dynamics import Dynamics
    from tvbo.utils.yaml_loader import load_as_dict

    return Dynamics(**load_as_dict(spec))


def _weights():
    return np.array([[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]])


def _render(defname, **kwargs):
    from tvbo import templates

    return templates.lookup.get_template("_linear_response.py.mako").get_def(defname).render(**kwargs)


def test_noise_terms_reads_the_declared_per_state_amplitudes():
    assert noise_terms(_model()) == [0.02, 0.0]


def test_noise_terms_is_none_when_no_state_declares_noise():
    spec = _SPEC.replace("""    noise:
      additive: true
      parameters: {sigma: {value: 0.02}}
""", "")
    assert noise_terms(_model(spec)) is None


def test_observable_terms_unfolds_a_derived_variable_chain():
    """``obs = k*y + drive`` with ``drive = m*x`` differentiates to ``[m, k]``."""
    terms = observable_terms(_model(), "obs")
    assert [str(e) for e in terms["Hloc"]] == ["m", "k"]
    assert terms["Hcpl"].shape == (1, 1) and terms["Hcpl"][0, 0] == 0


def test_observable_terms_accepts_a_state_variable():
    terms = observable_terms(_model(), "y")
    assert [float(e) for e in terms["Hloc"]] == [0.0, 1.0]


def test_observable_terms_rejects_an_undeclared_name():
    with pytest.raises(KeyError):
        observable_terms(_model(), "not_a_variable")


def _exec_rendered(model, sigma, obs_name):
    """Render and execute the partials, returning the covariance function's value."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    jax.config.update("jax_enable_x64", True)

    ctx = linear_response_context(model)
    W = _weights()
    fp = np.zeros((ctx["n_sv"], W.shape[0]))
    params = {p.name: p.value for p in model.parameters.values()}

    src = _render("lr_vf", ctx=ctx) + _render("lr_jacobian", ctx=ctx)
    if obs_name is not None:
        src += _render("lr_observable", ctx=ctx, name="_H_t",
                       terms=observable_terms(model, obs_name))
    src += _render("lr_covariance", ctx=ctx, name="_cov_t", sigma=sigma,
                   return_="covariance", obs_fn="_H_t" if obs_name else None)

    import types as _types

    ns = {"jnp": jnp, "jax": jax, "_lr_fp": jnp.asarray(fp), "_lr_weights": jnp.asarray(W),
          "_lr_params": _types.SimpleNamespace(**params)}
    exec(compile(src, "<lr>", "exec"), ns)
    A = np.asarray(ns["_lr_jacobian"](jnp.asarray(fp), jnp.asarray(W), ns["_lr_params"]))
    return np.asarray(ns["_cov_t"](A)), A, ctx, params


def test_covariance_through_a_declared_observation_cascade():
    """``H Sigma H^T`` with a per-state ``Q``, against a scipy Lyapunov solve."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    model = _model()
    P, A, ctx, params = _exec_rendered(model, sigma=None, obs_name="obs")

    N = _weights().shape[0]
    # The symbolic Jacobian is checked against a finite difference elsewhere; here the
    # NumPy oracle assembles the same A so the test isolates Q and H.
    A_ref = network_jacobian(model, _weights(), np.zeros((2, N)), params)
    assert np.allclose(A, A_ref, atol=1e-12)

    Q = np.diag(np.concatenate([np.full(N, s ** 2) for s in ctx["noise"]]))
    sigma_full = scipy_linalg.solve_continuous_lyapunov(A_ref, -Q)
    H = np.hstack([params["m"] * np.eye(N), params["k"] * np.eye(N)])
    assert np.allclose(P, H @ sigma_full @ H.T, rtol=1e-9, atol=1e-14)


def test_declared_noise_does_not_leak_into_an_unforced_state():
    """A state variable that declares no noise contributes no ``Q`` of its own.

    Under the uniform ``Q`` this test would fail: the unforced second state would be
    driven at the same amplitude as the first, inflating the observable's variance.
    """
    scipy_linalg = pytest.importorskip("scipy.linalg")
    model = _model()
    P_declared, A, ctx, params = _exec_rendered(model, sigma=None, obs_name="obs")

    N = _weights().shape[0]
    Q_uniform = (0.02 ** 2) * np.eye(2 * N)
    H = np.hstack([params["m"] * np.eye(N), params["k"] * np.eye(N)])
    P_uniform = H @ scipy_linalg.solve_continuous_lyapunov(A, -Q_uniform) @ H.T
    assert not np.allclose(P_declared, P_uniform, rtol=1e-3)
    assert np.all(np.diag(P_declared) < np.diag(P_uniform))


def test_uniform_sigma_still_returns_the_first_state_block():
    """The pre-existing contract: an explicit ``sigma`` means a uniform Q, first block out."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    model = _model()
    P, A, _, _ = _exec_rendered(model, sigma=0.02, obs_name=None)

    N = _weights().shape[0]
    Q = (0.02 ** 2) * np.eye(2 * N)
    assert P.shape == (N, N)
    assert np.allclose(P, scipy_linalg.solve_continuous_lyapunov(A, -Q)[:N, :N],
                       rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("dt,expected", [(1.0, 0.1), (0.1, 0.1), (0.001, 0.001), (None, 0.1)])
def test_settle_step_never_exceeds_the_recipes_own_integration_step(dt, expected):
    """A settle step is in MODEL time units, so a fixed one is right for only one time unit.

    0.1 is a tenth of a millisecond for a millisecond model and a hundred milliseconds for a
    second-based one — past the stability boundary of a 10 ms inhibitory time constant, which
    makes the operating point diverge rather than settle. Capping at the recipe's own step
    leaves every millisecond-unit recipe exactly where it was.
    """
    from tvbo.templates.tvboptim.utils import _lr_analysis_spec

    spec = _lr_analysis_spec({}, _model(), {}, None, 1.0, dt)
    assert spec["settle_dt"] == expected
