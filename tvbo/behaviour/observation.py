#
# Module: behaviour/observation.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""What an ``Observation`` record does: load by file or database name, render and build its backend monitor, and draw itself."""

from __future__ import annotations


class ObservationBehaviour:
    """A declared observation model's factory constructors, backend rendering and plots, on both generated forms."""

    _OP_COLORS = {
        "kernel": "#dbeafe",
        "convolution": "#fef9c3",
        "callable": "#fee2e2",
        "projection": "#dcfce7",
        "temporal": "#fae8ff",
        "transform": "#f1f5f9",
        "identity": "#f8fafc",
    }
    """Face colour per structural operation type of a pipeline step, for the flowchart boxes, in the order `_classify_pipeline` ranks them: a kernel generates a function over time (an HRF), a convolution folds two signals, a callable is any other external function, a projection maps over the node dimension, a temporal step averages or subsamples along time, a transform is any other equation, and identity is a passthrough. Every type `_step_op_type` names is a key here, so a box is coloured by lookup and never by fallback."""

    @classmethod
    def from_file(cls, path: str):
        """Load an Observation from a YAML file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(str(path), target_class=cls)

    @classmethod
    def from_db(cls, name: str):
        """Load an Observation by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Observation", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List the observation models available in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("Observation")

    def render_code(self, format: str = "tvb"):
        """Generate backend code that creates this monitor.

        Args:
            format: Target backend. Currently ``"tvb"`` is supported.

        Returns:
            Executable Python code string.
        """
        if format != "tvb":
            raise ValueError(f"Format {format!r} not supported for Observation. Use 'tvb'.")

        from tvbo import templates
        from tvbo.codegen.templater import format_code

        # Wrap single observation as the template expects experiment.observations dict
        class _Ctx:
            observations = {str(self.name): self}

        template = templates.lookup.get_template("tvbo-tvb-observation.py.mako")
        rendered = template.render(experiment=_Ctx())
        return format_code(rendered)

    def execute(self, format: str = "tvb"):
        """Convert this observation to a backend monitor object.

        Args:
            format: Target backend. Currently ``"tvb"`` is supported.

        Returns:
            The configured ``tvb.simulator.monitors.Monitor`` instance.
        """
        if format != "tvb":
            raise ValueError(f"Format {format!r} not supported for Observation. Use 'tvb'.")

        code = self.render_code("tvb")
        ns = {}
        exec(code, ns)
        monitors = ns.get("monitors", [])
        if monitors:
            return monitors[0]
        raise RuntimeError("Template produced no monitors")

    @staticmethod
    def _step_op_type(step) -> str:
        """Return the structural operation type of a single pipeline step."""
        if getattr(step, "time_range", None):
            return "kernel"
        if getattr(step, "callable", None):
            fn = str(getattr(step.callable, "name", "") or "").lower()
            if "convolve" in fn:
                return "convolution"
            return "callable"
        if getattr(step, "equation", None):
            dim = str(getattr(step, "apply_on_dimension", "") or "").lower()
            if dim == "node":
                return "projection"
            if dim == "time":
                return "temporal"
            return "transform"
        return "transform"

    def _classify_pipeline(self) -> str:
        """Classify the whole observation by its dominant pipeline axiom.

        No steps is ``"identity"``; otherwise the first type in `_OP_COLORS`'s order that any step has — kernel, convolution, callable, projection, temporal — and ``"transform"`` when none does.
        """
        steps = list(self.pipeline or [])
        if not steps:
            return "identity"
        types = {self._step_op_type(s) for s in steps}
        return next((dominant for dominant in self._OP_COLORS if dominant in types), "transform")

    def plot(self, ax=None, **kwargs):
        """Plot a visual summary of this observation model.

        The plot type is derived purely from the pipeline structure:

        * **kernel step present** (step with ``time_range``): evaluates and plots the kernel function.
        * **all other cases**: draws an annotated pipeline flowchart where each box is tagged with its structural operation type, one of the keys of `_OP_COLORS`.

        Args:
            ax: Matplotlib axes to draw into. A new figure is returned when ``ax`` is ``None``.
            **kwargs: Forwarded to the underlying plot call.
        """
        import matplotlib.pyplot as plt

        return_fig = ax is None
        if return_fig:
            fig, ax = plt.subplots(figsize=(3, 2.5))

        obs_class = self._classify_pipeline()
        if obs_class == "kernel":
            self._plot_kernel(ax, **kwargs)
        else:
            self._plot_pipeline_flowchart(ax, **kwargs)

        ax.set_title(str(self.label or self.name))

        if return_fig:
            plt.close(fig)
            return fig

    def _plot_kernel(self, ax, **kwargs):
        """Evaluate and plot the first pipeline step that has a ``time_range``.

        The window is the one the step declares — its ``time_range``'s ``lo``, ``hi`` and ``step``, each either a number or the name of a value the step declares — rather than two argument names guessed at, so a kernel that calls its duration anything else still plots over its own span. The first sample is skipped, because a gamma HRF is singular at the origin.

        The kernel is parsed in its own vocabulary — the step's numeric parameters and arguments, and ``t`` — so an undeclared name reaches SymPy's own, which is what lets an HRF call ``gamma``. It is then a function of ``t`` alone: a symbol left free after substitution means an unresolved parameter, not a second axis, so it is reported rather than silently bound to the time values.
        """
        import numpy as np
        from sympy import Symbol, lambdify

        from tvbo.parse.symbols import SymbolContext
        from tvbo.utils import keyed_items

        kernel_step = next((s for s in (self.pipeline or []) if getattr(s, "time_range", None)), None)
        if kernel_step is None or kernel_step.equation is None:
            ax.text(0.5, 0.5, "No kernel step found", ha="center", va="center", transform=ax.transAxes)
            return

        declared = [
            *keyed_items(getattr(kernel_step.equation, "parameters", None), "parameters"),
            *keyed_items(getattr(kernel_step, "arguments", None), "arguments"),
        ]
        param_subs = {}
        for name, member in declared:
            value = _as_float(getattr(member, "value", None))
            if value is not None:
                param_subs[Symbol(str(name))] = value

        span = kernel_step.time_range
        lo = _declared_number(getattr(span, "lo", None), param_subs, 0.0)
        hi = _declared_number(getattr(span, "hi", None), param_subs, 20.0)
        dt = _declared_number(getattr(span, "step", None), param_subs, 0.004)
        if dt <= 0 or hi <= lo + dt:
            ax.text(0.5, 0.5, "Kernel window is empty", ha="center", va="center", transform=ax.transAxes)
            return
        t_vals = np.arange(lo + dt, hi, dt)

        t_sym = Symbol("t")
        try:
            scope = SymbolContext({str(k): k for k in param_subs}, t=t_sym)
            expr_sub = scope.parse(str(kernel_step.equation.rhs)).subs(param_subs)
            free = expr_sub.free_symbols
            if free - {t_sym}:
                raise ValueError(f"unresolved in the kernel: {', '.join(sorted(str(s) for s in free - {t_sym}))}")
            y = lambdify([t_sym], expr_sub, modules="numpy")(t_vals) if free else float(expr_sub) * np.ones_like(t_vals)
        except Exception:
            ax.text(0.5, 0.5, "Kernel evaluation failed", ha="center", va="center", transform=ax.transAxes)
            return

        y = np.asarray(y, dtype=float)
        peak = np.nanmax(np.abs(y))
        if peak > 0:
            y = y / peak

        ax.plot(t_vals, y, **kwargs)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("kernel (norm.)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    def _plot_pipeline_flowchart(self, ax, **kwargs):
        """Draw pipeline steps as a vertical flowchart, each box coloured by and tagged with its structural operation type, one of the keys of `_OP_COLORS`."""
        import matplotlib.patches as mpatches
        import numpy as np

        steps = list(self.pipeline or [])

        ax.set_axis_off()

        if not steps:
            label = str((self.class_reference.name if self.class_reference else None) or self.name)
            ax.text(
                0.5,
                0.5,
                f"{label}\n(identity)",
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=self._OP_COLORS["identity"], edgecolor="gray", linewidth=0.8),
            )
            return

        n = len(steps)
        y_positions = np.linspace(0.88, 0.08, n)
        box_h = min(0.13, 0.75 / n)
        box_w = 0.78

        for idx, (step, yc) in enumerate(zip(steps, y_positions, strict=True)):
            op = self._step_op_type(step)
            fc = self._OP_COLORS[op]
            label = str(
                getattr(step, "label", None) or getattr(step, "name", None) or getattr(step, "output", None) or f"step {idx}"
            )
            rect = mpatches.FancyBboxPatch(
                (0.5 - box_w / 2, yc - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.02",
                linewidth=0.8,
                edgecolor="#94a3b8",
                facecolor=fc,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.add_patch(rect)
            # step name (main line) + operation tag (smaller, below)
            ax.text(0.5, yc + 0.012, label, ha="center", va="center", fontsize=7, transform=ax.transAxes)
            ax.text(0.5, yc - 0.022, f"[{op}]", ha="center", va="center", fontsize=5, color="#64748b", transform=ax.transAxes)

            if idx < n - 1:
                gap_start = yc - box_h / 2
                gap_end = y_positions[idx + 1] + box_h / 2
                ax.annotate(
                    "",
                    xy=(0.5, gap_end),
                    xytext=(0.5, gap_start),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.8),
                )


def _as_float(value):
    """*value* as a float, or ``None`` when it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _declared_number(value, declared, default):
    """*value* as a number: a literal as itself, a name *declared* holds a value for as that value, anything else as *default*."""
    from sympy import Symbol

    number = _as_float(value)
    if number is not None:
        return number
    if value is None:
        return default
    named = declared.get(Symbol(str(value)))
    return default if named is None else named
