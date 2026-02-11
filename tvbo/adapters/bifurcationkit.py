# -*- coding: utf-8 -*-
"""BifurcationKit.jl backend adapter for SimulationExperiment.

Uses juliacall to execute generated BifurcationKit Julia code
and return BifurcationResult objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvbo.analysis.bifurcation import BifurcationResult
    from tvbo.export.experiment import SimulationExperiment


class BifurcationKitAdapter:
    """Adapter for running bifurcation analysis via BifurcationKit.jl.

    Unlike NetworkDynamics/MTK adapters, this does not inherit from
    BaseAdapter — bifurcation analysis operates on individual
    (Dynamics, Continuation) pairs rather than a full network context.
    """

    def __init__(self, experiment: "SimulationExperiment"):
        self.experiment = experiment

    def render_code(self, model=None, continuation=None, **kwargs) -> str:
        """Render BifurcationKit Julia code for a single continuation.

        Parameters
        ----------
        model : Dynamics, optional
            The dynamics model. Defaults to ``experiment.local_dynamics``.
        continuation : Continuation, optional
            The continuation spec. Defaults to first in experiment.
        **kwargs
            Extra context passed to the Mako template.
        """
        from tvbo import templates

        model = model or self.experiment.local_dynamics
        if continuation is None:
            conts = getattr(self.experiment, "continuations", None) or {}
            if conts:
                continuation = next(iter(conts.values()))

        template = templates.lookup.get_template(
            "tvbo-julia-BifurcationKit.jl.mako"
        )
        return template.render(
            model=model, continuation=continuation, **kwargs
        )

    def run(self, **kwargs) -> "BifurcationResult | dict[str, BifurcationResult]":
        """Run bifurcation analysis for each continuation in the experiment.

        Iterates over ``experiment.continuations``, resolves the dynamics
        model for each, renders BifurcationKit Julia code, executes it,
        and wraps the result in ``BifurcationResult`` objects.

        Returns
        -------
        BifurcationResult or dict[str, BifurcationResult]
            Single result if one continuation, dict if multiple.
        """
        from tvbo.analysis import BifurcationResult
        from tvbo.run.julia import extract_bifurcation_result, run_julia_code

        exp = self.experiment
        conts = getattr(exp, "continuations", None) or {}
        if not conts:
            raise ValueError(
                "No continuations defined. Add continuation specs via "
                "exp.continuations or load from a bifurcation YAML."
            )

        results = {}
        for name, cont in conts.items():
            model = self._resolve_dynamics(cont)
            code = self.render_code(model=model, continuation=cont, **kwargs)

            run_julia_code(code)

            br_obj = extract_bifurcation_result()
            bif_res = BifurcationResult(br=br_obj, model=model, **kwargs)

            if getattr(cont, "branches", None):
                bif_res.periodic_orbits = self._extract_periodic_orbits(
                    model, **kwargs
                )

            results[name] = bif_res

        if len(results) == 1:
            return next(iter(results.values()))
        return results

    # ── Private helpers ──────────────────────────────────────────────────

    def _resolve_dynamics(self, cont):
        """Resolve the Dynamics model for a continuation spec."""
        exp = self.experiment
        dyn_ref = getattr(cont, "dynamics", None)
        if dyn_ref and str(dyn_ref) in exp.dynamics:
            return exp.dynamics[str(dyn_ref)]
        if exp.local_dynamics is not None:
            return exp.local_dynamics
        raise ValueError(
            f"Cannot resolve dynamics for continuation. "
            f"dynamics='{dyn_ref}' not found in exp.dynamics."
        )

    def _extract_periodic_orbits(self, model, **kwargs) -> list:
        """Extract periodic orbit branches from Julia Main after execution."""
        from tvbo.adapters.julia import eval_with_auto_install
        from tvbo.analysis import BifurcationResult

        try:
            po = eval_with_auto_install("po_results")
            return [
                BifurcationResult(br=p, model=model, **kwargs)
                for p in po.branches
            ]
        except Exception:
            return []
