"""Exogenous stimuli for simulation experiments.

Provides the [`Stimulus`](#tvbo.classes.perturbation.Stimulus) class, which
turns declarative stimulus metadata (from the datamodel, the ontology, or a
YAML file) into backend code and executable stimulus functions, plus helpers to
convert ontology classes to metadata and to replay audio files as stimuli.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None
logger = logging.getLogger(__name__)




def _require_librosa():
    if librosa is None:
        raise ImportError(
            "Audio features require 'librosa'. Install with:\n  pip install tvbo[audio]\nOr: pip install librosa"
        )


import owlready2 as owl
from scipy.interpolate import UnivariateSpline
from sympy import Symbol, lambdify, pycode, sympify

from tvbo import templates
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.codegen import templater
from tvbo.ontology import owl as ontology, query
from tvbo.classes import equation as equations
from tvbo.classes.equation import (
    _clash1,
    conditionals2piecewise,
    convert_ifelse_to_np_where,
)


def class2metadata(ontoclass):
    """Build `Stimulus` metadata from an ontology stimulus class.

    Reads the class's defining equation and, if it uses `where`, rewrites it
    into sympy form. The class name (identifier) and definition become the
    stimulus label and description, and every descendant `Parameter` is added
    with its default value and definition.

    Args:
        ontoclass: An owlready2 stimulus class whose `value`, `definition` and
            parameter descendants describe the perturbation.

    Returns:
        A datamodel `Stimulus` populated with the equation and parameters read
        from the ontology class.
    """

    onto_eq = ontoclass.value.first()
    if "where" in onto_eq:
        onto_eq = equations.convert_numpy_where_to_sympy(onto_eq)

    metadata = tvbo_datamodel.Stimulus(label=ontoclass.name, description=ontoclass.definition.first())
    metadata.equation = tvbo_datamodel.Equation(rhs=onto_eq)
    parameters = ontology.intersection(
        ontoclass.descendants(include_self=False),
        ontology.onto.Parameter.descendants(include_self=False),
    )
    for p in parameters:
        pname = ontology.replace_suffix(p)
        metadata.parameters.update(
            {
                pname: tvbo_datamodel.Parameter(
                    name=pname,
                    value=p.defaultValue.first(),
                    description=p.definition.first(),
                )
            }
        )
    return metadata


def load_acoustic_stimulus_from_audiofile(file_path, sampling_rate=1000, duration="full"):
    """Load an audio file as a callable stimulus time course.

    Loads the waveform, resamples it to `sampling_rate`, normalises it to the
    `[-1, 1]` range, optionally truncates it to `duration`, and fits a smoothing
    spline over time (in milliseconds).

    Args:
        file_path: Path to the audio file to read (any format `librosa`
            supports).
        sampling_rate: Target sampling rate in Hz used for resampling and for
            converting sample indices to milliseconds.
        duration: `"full"` to keep the whole signal, or a length in
            milliseconds to truncate it to.

    Returns:
        A function of time (in milliseconds) that evaluates the interpolated,
        normalised audio amplitude and returns 0 outside the signal's span.

    Raises:
        ImportError: If the optional `librosa` dependency is not installed.
    """
    _require_librosa()
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Resample the signal to the target sampling rate
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

    # Normalize the signal to [-1, 1] range
    normalized_audio = resampled_audio / np.max(np.abs(resampled_audio))

    audio = normalized_audio if duration == "full" else normalized_audio[: int(duration / 1000 * sampling_rate)]

    t = np.arange(len(audio)) / sampling_rate * 1000  # in ms
    audio_spline = UnivariateSpline(t, audio, s=0.01)

    def audio_fun(x):
        """Evaluate the audio spline at time `x`, returning 0 outside its span."""
        return (
            audio_spline(x)
            if np.isscalar(x) and t[0] <= x <= t[-1]
            else (np.where((x >= t[0]) & (x <= t[-1]), audio_spline(x), 0) if not np.isscalar(x) else 0)
        )

    return audio_fun


class Stimulus(tvbo_datamodel.Stimulus):
    """An exogenous perturbation injected into a `SimulationExperiment`.

    A `Stimulus` describes *what* gets perturbed (target state variable),
    *where* (spatial pattern across nodes), *when* (temporal envelope), and
    *how much* (amplitude). The pattern and envelope are arbitrary symbolic
    expressions, so the same class covers DC steps, sinusoids, Gaussian
    pulses, and audio-file replay.

    Attach via `experiment.add_stimulus(stim)`; see
    [`load_acoustic_stimulus_from_audiofile`](#tvbo.classes.perturbation.load_acoustic_stimulus_from_audiofile)
    for the WAV/MP3 entry point.
    """

    def __init__(self, **kwargs):
        # if isinstance(instance, owl.ThingClass):
        #     self.ontology = instance
        #     self = class2metadata(instance)
        # elif isinstance(instance, str) and (
        #     ontoclasses := query.label_search(
        #         instance,
        #         exact_match="all",
        #     )
        # ):
        #     if len(ontoclasses) > 1:
        #         print(f"Multiple stimulus classes found: {ontoclasses}")
        #     self.ontology = ontoclasses[0]
        #     self = class2metadata(self.ontology)
        # elif isinstance(instance, tvbo_datamodel.Stimulus):
        #     self = instance
        # else:
        #     if "name" not in kwargs:
        #         kwargs["name"] = "Stimulus"
        #     self = tvbo_datamodel.Stimulus(**kwargs)

        # if self.equation:
        #     eq, params = self.get_expression()
        if "label" not in kwargs:
            kwargs["label"] = kwargs.get("name", "Stimulus")
        super().__init__(**kwargs)

    @classmethod
    def from_datamodel(cls, instance: tvbo_datamodel.Stimulus):
        """Construct a `Stimulus` from a datamodel `Stimulus` instance.

        Args:
            instance: The datamodel stimulus whose fields are copied into the
                new `Stimulus`.

        Returns:
            A `Stimulus` initialised from the instance's fields.
        """
        return cls(**instance._as_dict)

    @classmethod
    def from_ontology(cls, ontoclass: str | owl.ThingClass):
        """Construct a `Stimulus` from an ontology class or its label.

        When given a string, searches the ontology for a stimulus class with
        that label (raising if none is found and warning if several match), then
        converts the resolved class to metadata via
        [`class2metadata`](#tvbo.classes.perturbation.class2metadata).

        Args:
            ontoclass: A stimulus label to look up, or an ontology stimulus
                class directly.

        Returns:
            A `Stimulus` populated from the ontology class.

        Raises:
            ValueError: If no stimulus class matches the given label.
        """
        if isinstance(ontoclass, str):
            ontoclasses = query.label_search(
                ontoclass,
                exact_match="all",
            )
            if not ontoclasses:
                raise ValueError(f"No stimulus class found for label '{ontoclass}'")
            if len(ontoclasses) > 1:
                logger.warning("Multiple stimulus classes found: %s", ontoclasses)
            ontoclass = ontoclasses[0]
        metadata = class2metadata(ontoclass)
        return cls(**metadata._as_dict)

    @classmethod
    def from_file(cls, filepath: os.PathLike):
        """Load a `Stimulus` from a YAML metadata file.

        Args:
            filepath: Path to the YAML file describing the stimulus.

        Returns:
            The `Stimulus` deserialised from the file.
        """
        from tvbo.utils import yaml_loader

        return yaml_loader.load(filepath, target_class=cls)

    @property
    def metadata(self):
        """The stimulus itself, exposed as its own metadata."""
        return self

    # @property
    # def equation(self):
    #     eq, params = self.get_expression()
    #     return eq

    def render_code(self, format="tvb", **kwargs):
        """Render the stimulus to backend source code.

        Selects the template for the requested backend, renders it with this
        stimulus, and formats the result.

        Args:
            format: Target backend: `"tvb"` for a TVB stimulus equation, or
                `"python"`/`"jax"` for a standalone stimulus function.
            **kwargs: Extra values forwarded to the template.

        Returns:
            The formatted source code as a string.
        """
        if format == "tvb":
            template = templates.lookup.get_template("tvbo-tvb-stimulus_equation.py.mako")
        elif format in ["python", "jax"]:
            template = templates.lookup.get_template("tvbo-python-stimulus.py.mako")
        rendered_code = template.render(stimulus=self, jax=format.lower() == "jax", **kwargs)
        return templater.format_code(rendered_code, format=format)

    def execute(
        self,
        format="tvb",
        connectivity=None,
        region_indices=None,
        weighting=None,
        **kwargs,
    ):
        """Build an executable stimulus for the requested backend.

        For `"tvb"`, evaluates the rendered stimulus equation, resolves a
        connectivity (creating a single-region one when needed) and a per-region
        weighting, and returns a TVB `StimuliRegion`. For `"python"`/`"jax"`,
        returns a callable stimulus function built from the symbolic equation,
        or from an audio file when the stimulus is defined by a `dataLocation`.

        Args:
            format: Target backend: `"tvb"`, `"python"`, or `"jax"`.
            connectivity: Connectivity for the TVB `StimuliRegion`; a one-region
                connectivity is created when omitted.
            region_indices: Indices of the stimulated regions; when omitted, a
                random permutation of all regions is used.
            weighting: Per-region weights; defaults to 1 on `region_indices` and
                0 elsewhere.
            **kwargs: Extra values forwarded to code rendering or audio loading
                (e.g. `sampling_rate`, `duration`).

        Returns:
            A TVB `StimuliRegion` for `"tvb"`, or a callable stimulus function
            for `"python"`/`"jax"`.
        """
        if format == "tvb":
            from tvb.datatypes.patterns import StimuliRegion

            namespace = {"exp": np.exp, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt}
            exec(self.render_code(), namespace)
            stim_eq = namespace[self.name + "Equation"]
            self.temporal_equation = stim_eq()

            if connectivity is None and format == "tvb":
                from tvbo.classes.network import Network

                sc = Network(number_of_regions=1)
                connectivity = sc.execute()

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

            return StimuliRegion(
                temporal=stim_eq(),
                connectivity=connectivity,
                weight=weighting,
            )

        if format in ["python", "jax"]:
            if self.equation:
                eq, param = self.get_expression()
                eq = eq.subs(param)
                code = self.render_code(format=format)
                namespace = {}
                exec(code, namespace)
                stim_func = namespace[self.label]
                # stim_func = lambdify("t", eq, modules="numpy")
            elif self.dataLocation:
                stim_func = load_acoustic_stimulus_from_audiofile(self.dataLocation, **kwargs)
            return stim_func

        elif format == "jax":
            eq, param = self.get_expression()
            return lambdify([Symbol("t")] + list(param.keys()), eq, modules="jax")

    def get_expression(self) -> tuple:
        """
        Generate a sympy expression for the equation using metadata.

        Returns:
            tuple: ``(expression, parameters)`` — the symbolic expression of the
            equation (or ``None``) and the resolved parameter substitution dict.
        """
        # Define symbols dynamically
        t = Symbol("t")
        params = {Symbol(k): v.value for k, v in self.parameters.items()}
        _clash1.update({"t": t})

        if self.equation is None:
            return None, params

        if self.equation.conditionals:
            eq = conditionals2piecewise(self.equation)

        # Parse the equation
        else:
            eq = sympify(self.equation.rhs, _clash1)

        if eq:
            self.function = lambdify("t", eq.subs(params), modules="numpy")
            python_code = pycode(eq, fully_qualified_modules=False)

        if self.equation.pycode:
            self.python_expression = self.equation.pycode
        else:
            self.python_expression = convert_ifelse_to_np_where(python_code) if "if" in python_code else python_code

        return eq, params

    def plot(self, duration=1000, dt=0.1, ax=None, plot_onset=True, cut_transient=0, **kwargs):
        """Plot the stimulus time course.

        Evaluates the python stimulus function over `[cut_transient, duration]`
        at step `dt` and draws it, optionally marking the `onset` parameter with
        a vertical line.

        Args:
            duration: End of the time window in milliseconds.
            dt: Sampling step in milliseconds.
            ax: Existing matplotlib axes to draw on; a new figure is created and
                returned when omitted.
            plot_onset: Whether to mark the `onset` parameter with a vertical
                line.
            cut_transient: Start of the time window in milliseconds.
            **kwargs: Extra keyword arguments forwarded to `ax.plot` (plus
                `sampling_rate`, used for stimulus evaluation).

        Returns:
            The created matplotlib figure when `ax` was not supplied; otherwise
            nothing.
        """
        t_ms = np.linspace(cut_transient, duration, int(duration / dt) + 1)

        stim_func = self.execute(
            format="python",
            duration=duration,
            sampling_rate=kwargs.pop("sampling_rate", 1000),
        )
        expr_values_ms = stim_func(t_ms)

        if ax is None:
            fig, ax = plt.subplots()
            return_fig = True
        else:
            return_fig = False

        ax.plot(t_ms, expr_values_ms, label="stimulus", **kwargs)

        if plot_onset and "onset" in self.parameters:
            ax.axvline(
                self.parameters["onset"].value,
                0,
                1,
                color="red",
                linestyle="--",
                label="onset",
            )

        if return_fig:
            plt.close()
            return fig
