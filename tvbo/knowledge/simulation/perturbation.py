import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import owlready2 as owl
from scipy.interpolate import UnivariateSpline
from sympy import Symbol, lambdify, pycode, sympify

from tvbo import templates
from tvbo.datamodel import tvbo_datamodel
from tvbo.export import templater
from tvbo.knowledge import ontology, query
from tvbo.knowledge.simulation import equations
from tvbo.knowledge.simulation.equations import (_clash1,
                                                 conditionals2piecewise,
                                                 convert_ifelse_to_np_where)


def class2metadata(ontoclass):

    onto_eq = ontoclass.value.first()
    if "where" in onto_eq:
        onto_eq = equations.convert_numpy_where_to_sympy(onto_eq)

    metadata = tvbo_datamodel.Stimulus(
        label=ontoclass.name, description=ontoclass.definition.first()
    )
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


def load_acoustic_stimulus_from_audiofile(
    file_path, sampling_rate=1000, duration="full"
):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Resample the signal to the target sampling rate
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

    # Normalize the signal to [-1, 1] range
    normalized_audio = resampled_audio / np.max(np.abs(resampled_audio))

    audio = (
        normalized_audio
        if duration == "full"
        else normalized_audio[: int(duration / 1000 * sampling_rate)]
    )

    t = np.arange(len(audio)) / sampling_rate * 1000  # in ms
    audio_spline = UnivariateSpline(t, audio, s=0.01)
    audio_fun = lambda x: (
        audio_spline(x)
        if np.isscalar(x) and t[0] <= x <= t[-1]
        else (
            np.where((x >= t[0]) & (x <= t[-1]), audio_spline(x), 0)
            if not np.isscalar(x)
            else 0
        )
    )
    return audio_fun


class Stimulus(tvbo_datamodel.Stimulus):
    def __init__(self, **kwargs):
        # if isinstance(instance, owl.ThingClass):
        #     self.ontology = instance
        #     self.metadata = class2metadata(instance)
        # elif isinstance(instance, str) and (
        #     ontoclasses := query.label_search(
        #         instance,
        #         exact_match="all",
        #     )
        # ):
        #     if len(ontoclasses) > 1:
        #         print(f"Multiple stimulus classes found: {ontoclasses}")
        #     self.ontology = ontoclasses[0]
        #     self.metadata = class2metadata(self.ontology)
        # elif isinstance(instance, tvbo_datamodel.Stimulus):
        #     self.metadata = instance
        # else:
        #     if "name" not in kwargs:
        #         kwargs["name"] = "Stimulus"
        #     self.metadata = tvbo_datamodel.Stimulus(**kwargs)

        # if self.metadata.equation:
        #     eq, params = self.get_expression()
        if "label" not in kwargs:
            kwargs["label"] = kwargs.get("name", "Stimulus")
        super().__init__(**kwargs)


    @classmethod
    def from_datamodel(cls, instance: tvbo_datamodel.Stimulus):
        return cls(**instance._as_dict)

    @classmethod
    def from_ontology(cls, ontoclass: str | owl.ThingClass):
        if isinstance(ontoclass, str):
            ontoclasses = query.label_search(
                ontoclass,
                exact_match="all",
            )
            if not ontoclasses:
                raise ValueError(f"No stimulus class found for label '{ontoclass}'")
            if len(ontoclasses) > 1:
                print(f"Multiple stimulus classes found: {ontoclasses}")
            ontoclass = ontoclasses[0]
        metadata = class2metadata(ontoclass)
        return cls(**metadata._as_dict)

    @classmethod
    def from_file(cls, filepath: os.PathLike):
        from linkml_runtime.loaders import yaml_loader

        return yaml_loader.load(filepath, target_class=cls)
    @property
    def metadata(self):
        return self
    # @property
    # def equation(self):
    #     eq, params = self.get_expression()
    #     return eq

    def render_code(self, format="tvb", **kwargs):
        if format == "tvb":
            template = templates.lookup.get_template(
                "tvbo-tvb-stimulus_equation.py.mako"
            )
        elif format in ["python", "jax"]:
            template = templates.lookup.get_template("tvbo-python-stimulus.py.mako")
        rendered_code = template.render(
            stimulus=self, jax=format.lower() == "jax", **kwargs
        )
        return templater.format_code(rendered_code, format=format)

    def execute(
        self,
        format="tvb",
        connectivity=None,
        region_indices=None,
        weighting=None,
        **kwargs,
    ):
        if format == "tvb":
            from tvb.datatypes.patterns import StimuliRegion

            namespace = {"exp": np.exp, "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt}
            exec(self.render_code(), namespace)
            stim_eq = namespace[self.name + "Equation"]
            self.temporal_equation = stim_eq()

            if connectivity is None and format == "tvb":
                from tvbo.data.tvbo_data.connectomes import Connectome

                sc = Connectome(number_of_regions=1)
                connectivity = sc.execute()

            if region_indices is None:
                region_indices = np.random.choice(
                    np.arange(connectivity.number_of_regions),
                    size=connectivity.number_of_regions,
                    replace=False,
                )
            if self.weighting:
                weighting = np.array(self.weighting)
                print(weighting)
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
                stim_func = load_acoustic_stimulus_from_audiofile(
                    self.dataLocation, **kwargs
                )
            return stim_func

        elif format == "jax":
            eq, param = self.get_expression()
            return lambdify([Symbol("t")] + list(param.keys()), eq, modules="jax")

    def get_expression(self):
        """
        Generate a sympy expression for the equation using metadata.

        Returns:
            sympy.Expr: The symbolic expression of the equation.
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
            self.python_expression = (
                convert_ifelse_to_np_where(python_code)
                if "if" in python_code
                else python_code
            )

        return eq, params

    def plot(
        self, duration=1000, dt=0.1, ax=None, plot_onset=True, cut_transient=0, **kwargs
    ):
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
