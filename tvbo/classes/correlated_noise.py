"""Correlated-noise lowering: a declared covariance becomes a Wiener-increment mixer.

`Noise.covariance` states the second-order structure of the driving process across the axis named by `Noise.correlated_over` — mathematics, not a factorisation. Turning that statement into samples is a backend concern, and this module is tvbo's concrete implementation of it for the JAX/tvboptim path.

It is built the way tvbo extends every backend: tvboptim supplies the abstract framework (`NativeSolver`, its `step` contract), and tvbo emits a concrete implementation against it. :class:`CorrelatedNoiseSolver` wraps any native solver and mixes the increment before delegating, exactly as tvboptim's own `BoundedSolver` wraps one and clips after. Because every integration path — the codegen template, the in-process heterogeneous runner, homogeneous and grouped networks alike — funnels its increment through `solver.step`, one wrapper covers all of them and there is no second mechanism to keep in sync.

Mixing iid draws by a factor ``L`` with ``L Lᵀ = C`` yields increments with covariance ``C`` along the chosen axis. Which factor is used is deliberately invisible to the spec:
Cholesky when ``C`` is positive definite, a symmetric eigendecomposition when it is only positive semi-definite (a rank-deficient covariance is legitimate — it says fewer independent sources than elements).

The declared reading is that σ carries the amplitude and ``C`` the correlation, so the realised covariance is ``diag(σ) C diag(σ) dt`` — for a scalar σ, ``σ² dt C``.

That composition is ``diag(σ) L``, NOT ``L diag(σ)``. The two agree exactly when σ is uniform along the mixed axis, which is why a per-node covariance with a scalar amplitude is insensitive to the difference; they diverge when σ varies along that axis, and there the wrong order is not a small error but a silent loss of the process. With a rank-deficient ``C`` — one independent source shared by two states, say — ``L``'s surviving column is placed by the eigendecomposition, and multiplying by a σ that is zero on the states the column happens to land on annihilates the increment entirely.

So the amplitude is folded into the covariance (:func:`fold_amplitudes`) and the increment arrives at unit amplitude, leaving the mixer to apply ``L'`` alone. Conjugating instead — ``diag(σ) L diag(1/σ⁺)`` — looks equivalent and is not, for the same reason: it drops the draw components at the zero-σ indices, which is where the rank-deficient source lives.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["covariance_factor", "fold_amplitudes", "noise_mixer", "CorrelatedNoiseSolver"]

_AXIS = {"state": -2, "node": -1, "region": -1}
"""`Noise.correlated_over` names mapped onto axes of the [n_noise_states, n_nodes]
increment. `mode` is absent by design: a model's modes are folded into the state axis."""

# Symmetry and PSD are checked against the matrix's own scale, so the tolerance means the same thing for a covariance of order 1e-6 and one of order 1e6.
_SYM_RTOL = 1e-8
_PSD_RTOL = 1e-10


def covariance_factor(cov, *, name: str = "covariance") -> np.ndarray:
    """A factor ``L`` with ``L Lᵀ = C``, after validating that ``C`` is a covariance.

    Raises rather than silently repairing: a non-symmetric or indefinite matrix is a specification error, and letting it through would surface as NaNs deep inside a jitted scan, far from the declaration that caused it.

    Args:
        cov: Square, symmetric, positive semi-definite matrix.
        name: Label used in error messages (the declaring slot).

    Returns:
        Lower-triangular Cholesky factor when `cov` is positive definite, otherwise
        ``V sqrt(Λ)`` from a symmetric eigendecomposition.
    """
    C = np.asarray(cov, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"{name}: expected a square matrix, got shape {C.shape}.")

    scale = float(np.abs(C).max()) or 1.0
    asym = float(np.abs(C - C.T).max())
    if asym > _SYM_RTOL * scale:
        raise ValueError(
            f"{name}: not symmetric — max |C - Cᵀ| = {asym:.3e} against a peak "
            f"magnitude of {scale:.3e}. A covariance is symmetric by definition; "
            f"symmetrise it in the producer if that is what was meant."
        )
    C = 0.5 * (C + C.T)

    try:
        return np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        pass

    # Not positive definite. Legitimate when the covariance is genuinely rank-deficient (fewer independent sources than elements); an error when it is indefinite.
    evals, evecs = np.linalg.eigh(C)
    most_negative = float(evals.min())
    if most_negative < -_PSD_RTOL * scale:
        raise ValueError(
            f"{name}: not positive semi-definite — smallest eigenvalue "
            f"{most_negative:.3e} against a peak magnitude of {scale:.3e}. "
            f"A covariance cannot have negative eigenvalues."
        )
    return evecs * np.sqrt(np.clip(evals, 0.0, None))


def fold_amplitudes(cov, sigmas, *, name: str = "covariance") -> np.ndarray:
    """``diag(σ) C diag(σ)`` — the declared covariance carried at each element's amplitude.

    Folding the amplitude in here, and driving the increment at unit amplitude, is what makes the realised covariance ``diag(σ) C diag(σ)`` rather than ``L diag(σ²) Lᵀ``. The two coincide for uniform σ, so this is a no-op wherever σ does not vary along the correlated axis; where it does vary, it is the difference between the declared process and (for a rank-deficient ``C``) no process at all.

    Args:
        cov: The declared covariance, square in the correlated axis.
        sigmas: Per-element amplitude along that axis, same length.
        name: Label for error messages.

    Returns:
        The amplitude-carrying covariance, ready for :func:`covariance_factor`.
    """
    C = np.asarray(cov, dtype=float)
    s = np.asarray(sigmas, dtype=float).ravel()
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"{name}: expected a square matrix, got shape {C.shape}.")
    if s.size != C.shape[0]:
        raise ValueError(
            f"{name}: {C.shape[0]}x{C.shape[0]} covariance against {s.size} amplitudes; "
            f"the covariance must be square in the axis the amplitudes index."
        )
    return C * np.outer(s, s)


def _axis_position(axis: str) -> int:
    """The increment axis a `correlated_over` name indexes.

    Raises for a name that is not an axis of the increment. `mode` gets its own message because the schema's own vocabulary offers it and the failure would otherwise read as a typo: a multi-mode model carries its modes inside the state axis, so a modal covariance is declared over `state`, not over a mode axis that does not exist here.
    """
    key = str(axis)
    if key == "mode":
        raise NotImplementedError(
            "correlated_over='mode' is not supported: the per-step noise increment is "
            "[n_noise_states, n_nodes] and a model's modes are folded into the state "
            "axis, so there is no mode axis to index. Declare the covariance over "
            "'state' (modes within a state variable) or 'node'."
        )
    if key not in _AXIS:
        raise ValueError(f"correlated_over={key!r} is not an axis of the noise increment; expected one of {sorted(_AXIS)}.")
    return _AXIS[key]


def _mix_leaf(factor, xi, axis_pos: int):
    """Impose the covariance on one increment array; pass non-arrays through.

    An ODE step carries the scalar ``0.0`` as its increment, and a group that declares no noise carries an all-zero block; both must survive untouched.
    """
    import jax.numpy as jnp

    if jnp.ndim(xi) < 2:
        return xi
    n = xi.shape[axis_pos]
    if factor.shape[0] != n:
        raise ValueError(
            f"covariance is {factor.shape[0]}x{factor.shape[0]} but the noise increment "
            f"has {n} elements on that axis; the covariance must be square in the axis "
            f"named by `correlated_over`."
        )
    # Contract the factor's source index against the correlated axis, leaving every other axis (leading time/block axes, and the axis not being correlated) alone.
    subs = "...sb,ab->...sa" if axis_pos == -1 else "...bn,ab->...an"
    return jnp.einsum(subs, xi, jnp.asarray(factor))


def noise_mixer(factor, axis: str = "node"):
    """A ``mix(xi) -> xi'`` that imposes the declared covariance on iid draws.

    Args:
        factor: The ``L`` from :func:`covariance_factor`.
        axis: Which axis of a ``[..., n_noise_states, n_nodes]`` block the covariance
            indexes — a `DimensionType` name.

    Returns:
        A callable mixing the trailing ``[n_noise_states, n_nodes]`` block of an array
        whose leading axes (time, blocks) are left untouched.
    """
    axis_pos = _axis_position(axis)

    def mix(xi):
        return _mix_leaf(factor, xi, axis_pos)

    return mix


def CorrelatedNoiseSolver(base_solver, factor, axis: str = "node"):
    """Wrap a native solver so its Wiener increment carries a declared covariance.

    tvboptim owns the integration step; this is tvbo's concrete solver against that abstract contract, mirroring the backend's own `BoundedSolver` (delegate, then transform — here the increment on the way in rather than the state on the way out).
    Wrapping is the one place that works for every network shape, because the grouped and ungrouped scans both hand their increment to `solver.step`.

    Args:
        base_solver: The native solver to wrap.
        factor: A single ``L`` applied to every group's increment, or a mapping from
            group name to ``L`` so a heterogeneous network can drive different groups
            with different processes. A group absent from the mapping keeps its
            independent increment.
        axis: The `correlated_over` axis name.

    Returns:
        A solver instance delegating to `base_solver`.
    """
    from tvboptim.experimental.network_dynamics.solvers import NativeSolver

    axis_pos = _axis_position(axis)

    class _CorrelatedNoiseSolver(NativeSolver):
        """A native solver that mixes the noise increment before delegating."""

        def __init__(self, base_solver, factor):
            """Wrap `base_solver`, delegating its scan settings via the properties below.

            Skips `NativeSolver.__init__` so wrapping cannot silently drop a block size or gradient horizon — the idiom `BoundedSolver` uses.
            """
            self.base_solver = base_solver
            self.factor = factor

        @property
        def block_size(self):
            return self.base_solver.block_size

        @property
        def recompute_coupling_per_stage(self):
            return self.base_solver.recompute_coupling_per_stage

        @property
        def grad_horizon(self):
            return self.base_solver.grad_horizon

        @property
        def stage_time_centroid(self):
            return self.base_solver.stage_time_centroid

        def _mix(self, noise_sample):
            if not isinstance(self.factor, Mapping):
                import jax

                return jax.tree.map(lambda leaf: _mix_leaf(self.factor, leaf, axis_pos), noise_sample)
            if not hasattr(noise_sample, "items"):
                raise ValueError(
                    "a per-group covariance was declared but the solver received a "
                    "single ungrouped noise increment; declare one covariance instead."
                )
            mixed = noise_sample.copy()
            for name, group_factor in self.factor.items():
                if name in mixed:
                    mixed[name] = _mix_leaf(group_factor, mixed[name], axis_pos)
            return mixed

        def step(self, dynamics_fn, t, state, dt, params, noise_sample=0.0):
            """Integration step whose increment carries the declared covariance."""
            return self.base_solver.step(dynamics_fn, t, state, dt, params, self._mix(noise_sample))

    return _CorrelatedNoiseSolver(base_solver, factor)
