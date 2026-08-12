"""The declared covariance must be realised as ``diag(sigma) C diag(sigma)``.

`Noise.covariance` states correlation and `sigma` states amplitude, so the realised
increment covariance is ``diag(sigma) C diag(sigma)``. Composing them the other way round
(``L diag(sigma)``, i.e. scaling the draw before mixing) agrees exactly when sigma is
uniform along the mixed axis and silently diverges when it is not — and for a
rank-deficient covariance it does not merely distort the process, it removes it: the
surviving eigenvector is placed by the eigendecomposition, and multiplying by a sigma that
is zero on the states it lands on annihilates the increment.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvbo.classes.correlated_noise import covariance_factor, fold_amplitudes


def _common_mode_over_states():
    """One shared source driving two of six states — the BEI eq (11)/(12) case."""
    C = np.zeros((6, 6))
    C[:2, :2] = 1.0
    sigmas = np.array([0.01, 0.01, 0.0, 0.0, 0.0, 0.0])
    return C, sigmas


def test_realised_covariance_equals_diag_sigma_C_diag_sigma():
    C, sigmas = _common_mode_over_states()
    L = covariance_factor(fold_amplitudes(C, sigmas))
    assert np.allclose(L @ L.T, np.diag(sigmas) @ C @ np.diag(sigmas), atol=1e-18)


def test_scaling_before_mixing_would_annihilate_a_rank_deficient_process():
    """The defect this guards: `L diag(sigma)` gives exactly zero here."""
    C, sigmas = _common_mode_over_states()
    wrong = covariance_factor(C) @ np.diag(sigmas)
    assert np.abs(wrong @ wrong.T).max() == 0.0
    right = covariance_factor(fold_amplitudes(C, sigmas))
    assert np.abs(right @ right.T).max() == pytest.approx(1e-4, rel=1e-9)


def test_uniform_sigma_is_unchanged_by_the_fold():
    """Why a `node`-axis covariance with a scalar amplitude cannot be affected.

    With sigma uniform along the mixed axis the two orders are bitwise identical, so the
    wave-model path (`correlated_over: node`, scalar sigma) is safe by construction rather
    than only by testing.
    """
    C = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    s = np.full(3, 0.02)
    assert np.array_equal(covariance_factor(C) @ np.diag(s), np.diag(s) @ covariance_factor(C))
    assert np.allclose(
        covariance_factor(fold_amplitudes(C, s)) @ covariance_factor(fold_amplitudes(C, s)).T, np.diag(s) @ C @ np.diag(s)
    )


def test_fold_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="amplitudes"):
        fold_amplitudes(np.eye(3), np.array([1.0, 2.0]))


def test_node_axis_emit_is_unchanged_by_the_fold(tmp_path):
    """A `correlated_over: node` recipe must emit no fold and keep its own sigma."""
    tvbo = pytest.importorskip("tvbo")
    spec = tmp_path / "s.yaml"
    spec.write_text("""
key: T
experiments:
  - id: 1
    label: e
    dynamics:
      name: Osc
      output: [x]
      parameters: {a: {value: 1.0}}
      state_variables:
        x:
          equation: {rhs: '-a*x'}
          initial_value: 0.1
          noise:
            additive: true
            parameters: {sigma: {value: 0.25}}
            covariance: {name: covariance, value: [[1.0, 0.5], [0.5, 1.0]]}
            correlated_over: node
      number_of_modes: 1
    network: {number_of_nodes: 2}
    integration: {method: heun, step_size: 0.1, duration: 1.0, transient_time: 0.0, unit: s}
    execution: {backend: tvboptim}
""")
    code = tvbo.SimulationStudy.from_file(str(spec)).get_experiment(1).render_code("tvboptim")
    assert "fold_amplitudes(_covariance" not in code
    assert "noise_sigma: float = 0.25" in code
