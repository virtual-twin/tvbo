"""tvbo's wrappers of a tvboptim native solver, each delegating the integration step and transforming what goes in or comes out.

tvboptim owns the step (`NativeSolver.step`) and wraps it itself in `BoundedSolver`; tvbo wraps it the same way where a declaration asks for something tvboptim does not do. Every integration path funnels through `solver.step`, so one wrapper covers all of them. tvboptim is imported when a wrapper is first built, as a backend dependency.
"""

from __future__ import annotations

from functools import cache

__all__ = ["delegating_solver", "ModelEventSolver"]


@cache
def delegating_solver():
    """The base class of tvbo's solver wrappers: a `NativeSolver` holding *base_solver* and reading its scan settings from it.

    It skips `NativeSolver.__init__`, as `BoundedSolver` does, so wrapping cannot silently drop a block size or a gradient horizon.
    """
    from tvboptim.experimental.network_dynamics.solvers import NativeSolver

    class DelegatingSolver(NativeSolver):
        """A native solver that delegates its step and its scan settings to *base_solver*."""

        def __init__(self, base_solver):
            self.base_solver = base_solver

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

        def step(self, dynamics_fn, t, state, dt, params, noise_sample=0.0):
            return self.base_solver.step(dynamics_fn, t, state, dt, params, noise_sample)

    return DelegatingSolver


def ModelEventSolver(base_solver, apply):
    """Wrap a native solver so the events a model declares act after each step: ``apply(state, params)`` returns the stepped state with every event whose condition holds applied.

    The affect lands before the coupling history and the recorded output read the step's state, so both see it, as LEMS and Brian2 apply a reset within the step that crossed the threshold.
    """

    class _ModelEventSolver(delegating_solver()):
        """A native solver that applies a model's events to the state each step returns."""

        def step(self, dynamics_fn, t, state, dt, params, noise_sample=0.0):
            """Integration step, then the model's events on the stepped state."""
            next_state, auxiliaries = self.base_solver.step(dynamics_fn, t, state, dt, params, noise_sample)
            return apply(next_state, params), auxiliaries

    return _ModelEventSolver(base_solver)
