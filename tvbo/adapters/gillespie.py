"""Gillespie SSA backend — a finite-size stochastic realization of a mean-field rate model.

Runs a relaxation-type rate model as a finite birth-death process (Gillespie 1977). The model must have one *activity* state variable ``X`` obeying a relaxation equation ``tau*X' = -X + F(state)`` (a leak ``-X`` toward a gain ``F``); any remaining state variables are treated as slow internal variables that evolve deterministically between events. The activity becomes a discrete count ``n ≈ Omega*X`` where ``Omega`` is the van Kampen system size (``execution.system_size``): the number of discrete units per unit of ``X``. The rate equation is read as a birth propensity ``a+ = Omega * F / tau`` (the gain term) and a death propensity ``a- = n / tau`` (the leak term, since ``a- = Omega*X/tau``), and the slow variables integrate deterministically over each inter-event interval. Finite ``Omega`` is the sole source of noise; the deterministic mean field is recovered as ``Omega -> infinity``. Applicable to any single-activity Wilson-Cowan / Tsodyks-Markram type rate model — the birth/death split and the between-event ODEs are derived from the model's own equations, so nothing here is model-specific.

Reference: Cortes et al. (2013) PNAS 110(41):16610, SI §2 (Eq. S10/S11) and Fig 5.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from tvbo.adapters.base import BaseAdapter
from tvbo.parse.expression import parse_eq
from tvbo.utils import initial_value


class GillespieAdapter(BaseAdapter):
    """Run a mean-field rate `SimulationExperiment` as a finite-N birth-death process.

    The window is `BaseAdapter.get_integration_info`'s: ``step_size`` is the output sampling cadence (events are drawn in continuous time), ``transient_time`` is simulated ahead of the measured ``duration`` in the same realisation, and the recorded grid is on the measurement clock. The initial state is not recorded, as on every other backend: the settle's ``transient_time / step_size`` samples end at ``t = 0`` on ``.transient``, and ``.data`` opens one step later with ``duration / step_size`` samples.
    """

    # -- symbolic model → numpy callables (birth flux + slow-variable RHS) -------------
    def _compile(self, model):
        """Return (activity_name, sv_names, tau, birth_fn, slow_fns) from the model equations.

        Everything is derived from the model's own equations: derived variables are inlined, parameters substituted, and the activity's ``tau*X' = -X + F`` split into a birth flux ``F/tau`` (the gain) and a death flux ``X/tau`` (the leak).
        """
        sv_names = list(model.state_variables)
        params = {k: float(v.value) for k, v in model.parameters.items()}
        derived_names = list(model.derived_variables)
        allsyms = {n: sp.Symbol(n) for n in (*sv_names, *params, *derived_names)}

        # Inline derived variables into pure (state-variable, parameter) expressions.
        derived = {d: sp.sympify(parse_eq(model.derived_variables[d].equation, symbols=allsyms)) for d in derived_names}
        for _ in range(len(derived) + 1):  # resolve nested derived refs to a fixed point
            derived = {d: e.xreplace({sp.Symbol(k): v for k, v in derived.items() if k != d}) for d, e in derived.items()}
        derived_subs = {sp.Symbol(k): v for k, v in derived.items()}
        param_subs = {sp.Symbol(k): v for k, v in params.items()}

        def rhs(name):
            e = sp.sympify(parse_eq(model.state_variables[name].equation, symbols=allsyms))
            return e.xreplace(derived_subs).xreplace(param_subs)

        activity = list(model.output or sv_names)[0]
        a_sym = allsyms[activity]
        r_a = rhs(activity)
        leak = r_a.coeff(a_sym, 1)  # coefficient of the linear -X/tau relaxation term
        if leak == 0:
            raise ValueError(
                f"gillespie backend: activity variable {activity!r} has no linear "
                f"relaxation (-{activity}/tau) term; it needs a rate equation of the form "
                "tau*X' = -X + F(...)."
            )
        tau = -1.0 / float(leak)
        birth_flux = r_a + a_sym / tau  # = F/tau, the gain (death flux is X/tau)

        argsyms = [allsyms[n] for n in sv_names]
        birth_fn = sp.lambdify(argsyms, birth_flux, "numpy")
        slow_fns = {n: sp.lambdify(argsyms, rhs(n), "numpy") for n in sv_names if n != activity}
        return activity, sv_names, tau, birth_fn, slow_fns

    def run(self, **kwargs):
        """Integrate the experiment with the Gillespie SSA and return the trajectory as an :class:`ExperimentResult`."""
        import xarray as xr

        from tvbo.data.types import ExperimentResult, SimulationResult

        exp = self.experiment
        model = exp.dynamics
        exe = getattr(exp, "execution", None)
        omega = getattr(exe, "system_size", None)
        if omega is None:
            raise ValueError(
                "the 'gillespie' backend requires execution.system_size (the van Kampen "
                "system size Omega): the number of discrete units per unit of the activity "
                "variable."
            )
        omega = float(omega)
        seed = int(getattr(exe, "random_seed", 0) or 0)

        activity, sv_names, tau, birth_fn, slow_fns = self._compile(model)
        slow = [n for n in sv_names if n != activity]
        bounds = {}
        for n in sv_names:
            dom = getattr(model.state_variables[n], "domain", None)
            bounds[n] = (
                None if dom is None else getattr(dom, "lo", None),
                None if dom is None else getattr(dom, "hi", None),
            )

        window = self.get_integration_info()
        rec_dt, n_settle = window["dt"], window["n_transient"]

        state = {n: initial_value(model.state_variables[n]) for n in sv_names}
        n_count = int(round(omega * state[activity]))
        rng = np.random.default_rng(seed)

        nrec = n_settle + window["n_measured"]
        rec = np.full((nrec, len(sv_names)), np.nan)
        ri, t_rec, t = 0, rec_dt, 0.0
        idx_activity = sv_names.index(activity)

        def bounded(n, value):
            """*value* of slow variable *n* clamped into its declared domain."""
            lo, hi = bounds[n]
            value = value if lo is None else max(value, float(lo))
            return value if hi is None else min(value, float(hi))

        while ri < nrec:  # until the recorded grid, settle and measured window, is full
            state[activity] = n_count / omega
            args = [state[n] for n in sv_names]
            a_plus = omega * float(birth_fn(*args))
            a_minus = n_count / tau
            a0 = a_plus + a_minus
            dt = -np.log(rng.random()) / a0 if a0 > 0 else rec_dt
            rates = {n: float(slow_fns[n](*args)) for n in slow}
            # A grid time before the event sees the count the interval holds and the slow variables on their Euler line.
            while t_rec < t + dt and ri < nrec:
                for j, n in enumerate(sv_names):
                    rec[ri, j] = n_count / omega if j == idx_activity else bounded(n, state[n] + rates[n] * (t_rec - t))
                ri += 1
                t_rec += rec_dt
            for n in slow:
                state[n] = bounded(n, state[n] + rates[n] * dt)
            if a0 > 0:
                n_count += 1 if rng.random() < a_plus / a0 else -1
                n_count = max(n_count, 0)
            t += dt

        time = (np.arange(1, ri + 1) - n_settle) * rec_dt
        da = xr.DataArray(
            rec[:ri],
            dims=["time", "variable"],
            coords={"time": time, "variable": sv_names},
        )
        return ExperimentResult(
            integration=SimulationResult(data=da, n_transient=n_settle),
            source=exp,
            name=getattr(exp, "label", None),
        )
