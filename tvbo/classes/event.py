"""Runtime wrapper around the auto-generated :class:`tvbo_datamodel.Event`.

Adds an ``intelligent`` :meth:`Event.plot` for stimulus-type events. The signal
is built generically from the event's symbolic equation and its parameters,
mirroring the pattern used by :class:`tvbo.classes.dynamics.Dynamics` and
:class:`tvbo.classes.perturbation.Stimulus`.
"""
from __future__ import annotations

import numpy as np
from sympy import Symbol, lambdify

from tvbo.datamodel import schema as tvbo_datamodel


class Event(tvbo_datamodel.Event):
    """:class:`tvbo_datamodel.Event` plus user-facing helpers."""

    def _signal(self):
        """Return ``callable(t)`` for the event's signal.

        Generic: works for any ``event.equation.rhs`` expressed in terms of
        ``t`` and the event's own parameters.
        """
        # Imported here rather than at module top: this module is imported by
        # ``tvbo.datamodel`` to attach the Event helpers, and parse.expression
        # imports back from ``tvbo.datamodel.schema`` — a module-top import would
        # form an import cycle when parse.expression is imported first.
        from tvbo.parse.expression import parse_eq

        params = {name: Symbol(name) for name in (self.parameters or {})}
        params.setdefault("t", Symbol("t"))
        expr = parse_eq(self.equation, local_dict=params)
        subs = {Symbol(name): float(p.value)
                for name, p in (self.parameters or {}).items()
                if p.value is not None}
        return lambdify(Symbol("t"), expr.subs(subs), modules="numpy")

    def _default_window(self):
        """Infer a sensible (t0, t1) window from event parameters."""
        p = self.parameters or {}
        onset = float(p["onset"].value) if "onset" in p and p["onset"].value is not None else 0.0
        width = (float(p["width"].value) if "width" in p and p["width"].value is not None
                 else float(self.duration) if self.duration else 1.0)
        return onset - max(width, 0.1), onset + width + max(width, 0.1)

    def plot(self, t=None, n=1001, ax=None, onset_kw=None, **kwargs):
        """Plot the event signal vs time.

        Parameters
        ----------
        t : array-like, optional
            Time samples. If ``None``, inferred from event parameters.
        n : int
            Number of samples when ``t`` is auto-generated.
        ax : matplotlib Axes, optional
            Axes to draw into. A new figure is returned when ``ax`` is ``None``.
        onset_kw : dict, optional
            Style overrides for the onset vertical line, e.g.
            ``onset_kw={"color": "blue", "linewidth": 2}``.
            Defaults: ``{"color": "red", "linestyle": "--", "linewidth": 0.7}``.
        **kwargs
            Forwarded to ``ax.plot`` for the signal line.
        """
        import matplotlib.pyplot as plt

        if t is None:
            t0, t1 = self._default_window()
            t = np.linspace(t0, t1, n)
        t = np.asarray(t, dtype=float)

        f = self._signal()
        y = np.broadcast_to(np.asarray(f(t), dtype=float), t.shape)

        return_fig = ax is None
        if return_fig:
            fig, ax = plt.subplots()
        ax.plot(t, y, label=str(self.name), **kwargs)
        ax.set_xlabel("t")
        ax.set_ylabel(str(self.name))
        if "onset" in (self.parameters or {}):
            onset = float(self.parameters["onset"].value)
            _onset_kw = {"color": "red", "linestyle": "--", "linewidth": 0.7, "label": "onset"}
            if onset_kw:
                _onset_kw.update(onset_kw)
            ax.axvline(onset, **_onset_kw)
        ax.legend(loc="best")
        if return_fig:
            plt.close(fig)
            return fig
        return ax


# Make the helpers available on the auto-generated schema class itself, so
# ``schema.Event(...).plot()`` works without requiring callers to import the
# wrapper explicitly. This mirrors the ``__class__`` patching pattern used for
# Network/Continuation in :mod:`tvbo.classes.experiment`.
for _name in ("plot", "_signal", "_default_window"):
    setattr(tvbo_datamodel.Event, _name, getattr(Event, _name))

# ``regions`` -> ``nodes`` and ``weighting`` -> ``weights`` are declared aliases; the
# generated dialect (:mod:`tvbo.datamodel.dialect`) folds them for every class, so the
# hand-written copy that used to sit here is gone.
