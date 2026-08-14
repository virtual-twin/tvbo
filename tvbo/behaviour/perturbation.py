"""Symbolic reading, code generation and plotting for :class:`Stimulus`.

Attached to the generated classes by name (``StimulusBehaviour`` -> ``Stimulus``), so a
stimulus carries these however it was built — loaded from YAML, nested in an experiment,
or constructed directly. The experiment loader used to reassign ``__class__`` to reach
the nested one, which left every other construction path without them.

A stimulus is emitted as a Python definition, and the only name it has is ``label``, which
is free text. :attr:`StimulusBehaviour.identifier` is that one resolution — the templates
and :meth:`StimulusBehaviour.execute` both read it, so what is emitted and what is looked
up afterwards cannot drift apart.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class StimulusBehaviour:
    """Loading, symbolic reading, rendering and plotting for an exogenous stimulus."""

    @classmethod
    def from_datamodel(cls, instance):
        """Copy a datamodel `Stimulus`'s fields into a new one."""
        return cls(**instance._as_dict)

    @classmethod
    def from_ontology(cls, ontoclass):
        """Construct a `Stimulus` from an ontology class or its label.

        When given a string, searches the ontology for a stimulus class with that label,
        raising if none is found and warning if several match, then converts the resolved
        class to metadata via
        [`class2metadata`](#tvbo.classes.perturbation.class2metadata).

        Args:
            ontoclass: A stimulus label to look up, or an ontology stimulus class.

        Returns:
            A `Stimulus` populated from the ontology class.

        Raises:
            ValueError: If no stimulus class matches the given label.
        """
        from tvbo.classes.perturbation import class2metadata
        from tvbo.ontology import query

        if isinstance(ontoclass, str):
            ontoclasses = query.label_search(ontoclass, exact_match="all")
            if not ontoclasses:
                raise ValueError(f"No stimulus class found for label '{ontoclass}'")
            if len(ontoclasses) > 1:
                logger.warning("Multiple stimulus classes found: %s", ontoclasses)
            ontoclass = ontoclasses[0]
        return cls(**class2metadata(ontoclass)._as_dict)

    @classmethod
    def from_file(cls, filepath):
        """Load a `Stimulus` from a YAML metadata file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(filepath, target_class=cls)

    @property
    def metadata(self):
        """The stimulus itself, exposed as its own metadata."""
        return self

    @property
    def identifier(self) -> str:
        """The Python name the stimulus is emitted under.

        `label` is free text, so it is not a name a backend can be handed as one. This
        turns it into an identifier once, for every backend and for the lookup that
        follows execution.
        """
        from tvbo.templates.base.utils import safe_name

        return safe_name(getattr(self, "label", None), fallback="Stimulus")

    def render_code(self, format="tvb", **kwargs):
        """Render the stimulus to backend source code.

        Args:
            format: Target backend: `"tvb"` for a TVB stimulus equation, or
                `"python"`/`"jax"` for a standalone stimulus function.
            **kwargs: Extra values forwarded to the template.

        Returns:
            The formatted source code as a string.
        """
        from tvbo import templates
        from tvbo.codegen import templater

        fmt = format.lower()
        name = {
            "tvb": "tvbo-tvb-stimulus_equation.py.mako",
            "python": "tvbo-python-stimulus.py.mako",
            "jax": "tvbo-python-stimulus.py.mako",
        }[fmt]
        rendered_code = templates.lookup.get_template(name).render(stimulus=self, jax=fmt == "jax", **kwargs)
        return templater.format_code(rendered_code, format=format)

    def execute(self, format="tvb", connectivity=None, region_indices=None, weighting=None, **kwargs):
        """Build an executable stimulus for the requested backend.

        For `"tvb"`, evaluates the rendered stimulus equation, resolves a connectivity
        (creating a single-region one when needed) and a per-region weighting, and returns
        a TVB `StimuliRegion`. For `"python"`/`"jax"`, returns a callable stimulus function
        built from the symbolic equation, or from an audio file when the stimulus is
        defined by a `dataLocation`.

        Args:
            format: Target backend: `"tvb"`, `"python"`, or `"jax"`.
            connectivity: Connectivity for the TVB `StimuliRegion`; a one-region
                connectivity is created when omitted.
            region_indices: Indices of the stimulated regions; when omitted, a random
                permutation of all regions is used.
            weighting: Per-region weights; defaults to 1 on `region_indices` and 0
                elsewhere.
            **kwargs: Extra values forwarded to code rendering or audio loading (e.g.
                `sampling_rate`, `duration`).

        Returns:
            A TVB `StimuliRegion` for `"tvb"`, or a callable stimulus function for
            `"python"`/`"jax"`.
        """
        if format.lower() == "tvb":
            from tvb.datatypes.patterns import StimuliRegion

            namespace = {"exp": np.exp, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt}
            exec(self.render_code(), namespace)
            stim_eq = namespace[self.identifier + "Equation"]

            if connectivity is None:
                from tvbo.classes.network import Network

                connectivity = Network(number_of_regions=1).execute()

            if region_indices is None:
                region_indices = np.random.choice(
                    np.arange(connectivity.number_of_regions),
                    size=connectivity.number_of_regions,
                    replace=False,
                )
            if self.weighting:
                weighting = np.array(self.weighting)
                logger.debug("stimulus weighting: %s", weighting)
            elif weighting is None and connectivity:
                weighting = np.zeros(connectivity.number_of_regions)
                weighting[region_indices] = 1

            return StimuliRegion(temporal=stim_eq(), connectivity=connectivity, weight=weighting)

        if self.equation:
            namespace = {}
            exec(self.render_code(format=format), namespace)
            return namespace[self.identifier]

        from tvbo.classes.perturbation import load_acoustic_stimulus_from_audiofile

        return load_acoustic_stimulus_from_audiofile(self.dataLocation, **kwargs)

    def get_expression(self) -> tuple:
        """Generate a sympy expression for the equation using metadata.

        The one place a stimulus is parsed: the equation's parameters shadow the SymPy
        builtins they collide with, and `t` is the time symbol whichever spelling the
        metadata used — a right-hand side or a list of conditional branches.

        Returns:
            tuple: ``(expression, parameters)`` — the symbolic expression of the equation
            (or ``None``) and the resolved parameter substitution dict.
        """
        from sympy import Symbol

        from tvbo.parse.symbols import BUILTIN_SHADOW

        params = {Symbol(k): v.value for k, v in self.parameters.items()}
        if self.equation is None:
            return None, params
        scope = BUILTIN_SHADOW.extend({str(p): p for p in params}, t=Symbol("t"))
        return scope.parse(self.equation), params

    def plot(self, duration=1000, dt=0.1, ax=None, plot_onset=True, cut_transient=0, **kwargs):
        """Plot the stimulus time course.

        Evaluates the python stimulus function over `[cut_transient, duration]` at step
        `dt` and draws it, optionally marking the `onset` parameter with a vertical line.

        Args:
            duration: End of the time window in milliseconds.
            dt: Sampling step in milliseconds.
            ax: Existing matplotlib axes to draw on; a new figure is created and returned
                when omitted.
            plot_onset: Whether to mark the `onset` parameter with a vertical line.
            cut_transient: Start of the time window in milliseconds.
            **kwargs: Extra keyword arguments forwarded to `ax.plot` (plus
                `sampling_rate`, used for stimulus evaluation).

        Returns:
            The created matplotlib figure when `ax` was not supplied; otherwise nothing.
        """
        import matplotlib.pyplot as plt

        t_ms = np.linspace(cut_transient, duration, int(duration / dt) + 1)
        stim_func = self.execute(format="python", duration=duration, sampling_rate=kwargs.pop("sampling_rate", 1000))
        expr_values_ms = stim_func(t_ms)

        return_fig = ax is None
        if return_fig:
            fig, ax = plt.subplots()

        ax.plot(t_ms, expr_values_ms, label="stimulus", **kwargs)

        if plot_onset and "onset" in self.parameters:
            ax.axvline(self.parameters["onset"].value, 0, 1, color="red", linestyle="--", label="onset")

        if return_fig:
            plt.close()
            return fig
