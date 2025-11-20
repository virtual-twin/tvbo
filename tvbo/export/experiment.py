#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
import copy as _copy
import os
from os.path import join

import jax
import jax.numpy as jnp
import numpy as np
from lems.base.util import validate_lems

from tvbo import templates
from tvbo.data.tvbo_data.connectomes import Connectome
from tvbo.data.types import SimulationState, TimeSeries
from tvbo.datamodel import tvbo_datamodel
from tvbo.export import templater
from tvbo.export.templater import format_code
from tvbo.knowledge import Connectome, Coupling, Integrator
from tvbo.knowledge.simulation.localdynamics import Dynamics
from tvbo.knowledge.simulation.network import Coupling, Network
from tvbo.parse import metadata
from tvbo.utils import Bunch

sessionid = 1

class SimulationExperiment(tvbo_datamodel.SimulationExperiment):
    def __init__(self, **kwargs):
        """Initialize like the datamodel, but auto-assign an id when missing.

        Supports any of the following inputs:
        - A tvbo_datamodel.SimulationExperiment instance
        - A dict of fields
        - Keyword args matching the datamodel fields
        """
        global sessionid

        # Ensure an id exists (the datamodel requires it in __post_init__)
        if kwargs.get("id") is None:
            kwargs["id"] = sessionid
            sessionid += 1

        # Delegate to the parent dataclass initializer for normalization
        super().__init__(**kwargs)

        # Normalize and coerce fields while preserving original conditions
        def _coerce(cls, obj):
            if isinstance(obj, cls):
                return obj
            if hasattr(obj, "_as_dict"):
                return cls(**obj._as_dict)
            if isinstance(obj, dict):
                return cls(**obj)
            return obj

        # Prefer `model` when `local_dynamics` is missing
        if getattr(self, "model", None) and not getattr(self, "local_dynamics", None):
            self.local_dynamics = self.model

        # Ensure proper types
        if getattr(self, "local_dynamics", None) and not isinstance(
            self.local_dynamics, Dynamics
        ):
            self.local_dynamics = _coerce(Dynamics, self.local_dynamics)

        if getattr(self, "coupling", None) and not isinstance(self.coupling, Coupling):
            self.coupling = _coerce(Coupling, self.coupling)

        if getattr(self, "integration", None) and not isinstance(
            self.integration, Integrator
        ):
            self.integration = _coerce(Integrator, self.integration)

        # Backward-compat aliasing for connectivity/network
        if getattr(self, "connectivity", None) and not getattr(self, "network", None):
            self.network = self.connectivity

        if getattr(self, "network", None) and not isinstance(self.network, Connectome):
            self.network = _coerce(Connectome, self.network)
            self.connectivity = self.network  # backwards compatibility TODO: remove

        # Mirror model/local_dynamics
        self.model = self.local_dynamics

        # Defaults
        if not getattr(self, "monitors", None):
            from tvbo.datamodel.tvbo_datamodel import Monitor

            self.monitors["Raw"] = Monitor(name="Raw")

        if not getattr(self, "network", None):
            self.network = Connectome()

        if not getattr(self, "integration", None):
            self.integration = Integrator(method="Heun")

        if not getattr(self, "coupling", None):
            self.coupling = Coupling(name="Linear")

    @classmethod
    def from_datamodel(
        cls, dm: tvbo_datamodel.SimulationExperiment
    ) -> "SimulationExperiment":
        # Leverage the unified initializer
        return cls(**dm._as_dict)

    @classmethod
    def from_tvb_simulator(cls, tvb_simulator):
        return cls.from_datamodel(metadata.simulator2metadata(tvb_simulator))

    @classmethod
    def from_file(cls, filepath: str):
        from linkml_runtime.loaders import yaml_loader

        return yaml_loader.load(filepath, target_class=cls)

    @property
    def metadata(self):
        return self

    @property
    def noise_sigma_array(self) -> np.ndarray:
        """Per-state-variable noise sigma values.

        Preference order:
        1) sigma from each state variable's noise.parameters["sigma"].value
        2) fallback to integration-level noise.parameters["sigma"].value
        3) default 0.0

        Returns an array with one entry per state variable in model order.
        """
        sigmas: list[float] = []

        for sv in self.local_dynamics.state_variables.values():
            sigma = 0.0
            if sv.noise:
                try:
                    sigma = float(sv.noise.parameters["sigma"].value)
                except Exception as e:
                    print(f"Error retrieving sigma for state variable {sv.name}: {e}")
                    pass

            if sigma == 0.0:
                try:
                    integ_meta = getattr(self.integration, "metadata", None)
                    integ_noise = getattr(integ_meta, "noise", None)
                    inparams = getattr(integ_noise, "parameters", None)
                    if (
                        inparams is not None
                        and isinstance(inparams, dict)
                        and "sigma" in inparams
                        and hasattr(inparams["sigma"], "value")
                    ):
                        sigma = float(inparams["sigma"].value)
                except Exception as e:
                    print(f"Error retrieving integration-level sigma: {e}")
                    pass

            sigmas.append(float(sigma))

        if np.any(np.asarray(sigmas, dtype=float) > 0):
            self.integration.state_wise_sigma = sigmas
            if not self.integration.noise:
                from tvbo.knowledge.simulation.integration import Noise

                self.integration.noise = Noise()

        return np.asarray(sigmas, dtype=float)

    def __str__(self):
        return self.label if self.label else f"SimulationExperiment{self.id}"

    def __repr__(self):
        return self.__str__()

    # ---- Copy utilities ----
    def copy(self, **overrides) -> "SimulationExperiment":
        """Return a deep copy of this experiment.

        Use keyword overrides to set attributes on the returned copy.

        Errors are not swallowed; if a field can't be copied, an exception is raised.
        """
        new_obj = _copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(new_obj, k, v)
        return new_obj

    # Python copy protocol hooks
    def __copy__(self):
        # Keep Python's copy.copy semantics: shallow copy
        cls = self.__class__
        clone = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(clone, k, v)
        return clone

    def __deepcopy__(self, memo):
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for k, v in self.__dict__.items():
            setattr(clone, k, _copy.deepcopy(v, memo))
        return clone

    def to_yaml(self, filepath: str | None = None):
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, filepath)

    def render_yaml(self) -> str:
        """Deprecated Render the YAML representation as a string.
        Use to_yaml(filepath=None) instead.
        """
        return self.to_yaml(filepath=None)

    def setup_monitors(self, **kwargs):
        """Populate monitors in metadata from simple inputs or runtime wrappers."""
        from tvbo.knowledge.simulation import monitoring

        monitors_ = kwargs.get("monitors", [])
        meta_list = []
        if isinstance(monitors_, monitoring.Monitor):
            meta_list = [monitors_.metadata]
        else:
            for m in monitors_:
                if isinstance(m, monitoring.Monitor):
                    meta_list.append(m.metadata)
                elif isinstance(m, dict):
                    meta_list.append(tvbo_datamodel.Monitor(**m))
        if meta_list:
            self.monitors = meta_list

    def configure(self):
        # Disable delayed logic if the connectome has no path lengths
        try:
            conn = (
                Connectome(self.network)
                if getattr(self, "network", None) is not None
                else None
            )
            L = conn.lengths_matrix if conn is not None else None
            if L is not None and np.allclose(L, 0):
                if getattr(self, "integration", None) is not None:
                    self.integration.delayed = False
                if getattr(self, "coupling", None) is not None:
                    self.coupling.delayed = False
        except Exception:
            # Best-effort; keep defaults if anything goes wrong
            pass

    def add_stimulus(self, stimulus):
        import owlready2 as owl

        from tvbo.knowledge.simulation import perturbation

        if isinstance(stimulus, perturbation.Stimulus):
            self.stimulation = stimulus
        elif isinstance(stimulus, str) or isinstance(stimulus, owl.ThingClass):
            self.stimulation = perturbation.Stimulus.from_ontology(stimulus)

    def collect_state(self, initial_conditions: TimeSeries | None = None):
        _ = self.noise_sigma_array
        state = SimulationState(
            initial_conditions=(
                initial_conditions
                if initial_conditions is not None
                else self.collect_initial_conditions()
            ),
            network=self.network,
            dt=self.integration.step_size,
            nt=int(np.ceil(self.integration.duration / self.integration.step_size)),
            # Provide a JAX-pytree-friendly Noise wrapper (or None)
            noise=self.integration.noise_wrapper,
            parameters=self.get_parameters_collection(
                keys_to_exclude=[
                    "derived_parameters",
                    "conduction_speed",
                    "coupling_terms",
                ]
            ),
            stimulus=None,
            monitor_parameters=None,
        )
        # Attach state variable names for ergonomic noise setters
        try:
            state._svar_names = list(self.local_dynamics.state_variables.keys())
        except Exception:
            pass
        return state

    def execute(self, format="tvb", **kwargs):
        if format.lower() == "tvb":
            code = self.render_code(format=format)
            namespace = templater.exec_globals
            exec(code, namespace)
            sim = namespace["define_simulation"](
                connectivity=self.network.execute("tvb"), **kwargs
            )
            sim.initial_conditions = self.collect_initial_conditions().data
            sim.configure()
            return sim

        elif format.lower() in ["autodiff", "jax"]:
            jit = kwargs.get("jit", True)
            code = self.render_code(format=format, **kwargs)
            namespace = templater.exec_globals
            namespace.update({"TimeSeries": TimeSeries})
            exec(code, namespace)
            jax_model = namespace["kernel"]
            if jit:
                jax_model = jax.jit(jax_model)
            return jax_model

        elif format.lower() in ["pde", "pde-fem", "pde-python"]:
            code = self.render_code(format="pde")
            namespace = templater.exec_globals
            exec(code, namespace)
            return namespace

        else:
            raise ValueError(f"Format {format} not supported. Valid formats: tvb, jax.")

    def run(self, format="jax", initial_conditions=None, **kwargs):
        self.configure()
        simulation_data = Bunch()

        if format.lower() == "tvb":
            initial_conditions = self.collect_initial_conditions(
                random=kwargs.pop("random_initial_conditions", False)
            )
            simulator_ = self.execute()
            simulator_.initial_conditions = initial_conditions.data
            simulator_.configure()
            simres = simulator_.run(**kwargs)
            derivatives = []
            labels_dim = {
                "State Variable": simulator_.model.variables_of_interest,
                "Region": list(simulator_.connectivity.region_labels),
            }
            for m, (tv, xv) in zip(simulator_.monitors, simres):
                m_name = m.title.split(" ")[0]
                if m_name == "Raw":
                    ts = TimeSeries(
                        data=xv,
                        time=tv,
                        labels_dimensions=labels_dim,
                        title=m_name,
                        sample_period=m.period,
                    )
                else:
                    derivatives.append(
                        TimeSeries(
                            data=xv,
                            time=tv,
                            labels_dimensions=labels_dim,
                            title=m_name,
                            sample_period=m.period,
                        )
                    )
            ts.derivatives = derivatives
            return ts

        elif format.lower() in ["autodiff", "jax"]:
            state = self.collect_state(initial_conditions=initial_conditions)
            if kwargs.get("enable_x64", True):
                jax.config.update("jax_enable_x64", True)
                state = state.convert_dtype(target_dtype=jnp.float64)
            else:
                jax.config.update("jax_enable_x64", False)
                state = state.convert_dtype(target_dtype=jnp.float32)

            jax_model = self.execute(format="jax", **kwargs)
            ts = jax_model(state)
            # simulation_data = Bunch()
            # ts.labels_dimensions = {
            #     "State Variable": list(self.local_dynamics.state_variables.keys()),
            #     "Region": self.network.labels,
            # }
            # ts.sample_period = self.integration.step_size
            # ts.dt = self.integration.step_size

            return ts

        elif format.lower() == "python":
            bnm = Network(Connectome(self.network))
            bnm.add_local_model(self.local_dynamics)
            bnm.add_coupling(self.coupling)

            ts = bnm.run(
                duration=kwargs.get("duration", self.integration.duration),
                dt=self.integration.step_size,
            )
            simulation_data["Raw"] = ts

        elif format.lower() in ["pde", "pde-fem", "pde-python"]:
            ns = self.execute(format="pde")
            solve = ns.get("solve_pde")
            viz = ns.get("visualize")
            meta = ns.get("meta")

            if solve is None:
                raise RuntimeError("PDE backend did not expose solve_pde.")

            steps = kwargs.get("steps", None)
            out = kwargs.get("out", None)

            # Optional node-based initial condition and source
            u0_override = None
            src = kwargs.get("source", None)

            # Accept explicit u0 or u0_override kwarg
            u0_kw = kwargs.get("u0", None)
            if u0_kw is None:
                u0_kw = kwargs.get("u0_override", None)

            if u0_kw is not None:
                arr = np.asarray(u0_kw, dtype=float).ravel()
                if meta and isinstance(meta, dict):
                    ndofs = int(meta.get("ndofs", arr.size))
                    if arr.size != ndofs:
                        raise ValueError(f"u0 length {arr.size} != ndofs {ndofs}")
                u0_override = arr
            elif initial_conditions is not None:
                # Support TimeSeries or ndarray as initial conditions
                if isinstance(initial_conditions, TimeSeries):
                    arr = np.asarray(
                        initial_conditions.data[-1, 0, :, 0], dtype=float
                    ).ravel()
                else:
                    arr = np.asarray(initial_conditions, dtype=float).ravel()
                if meta and isinstance(meta, dict):
                    ndofs = int(meta.get("ndofs", arr.size))
                    if arr.size != ndofs:
                        raise ValueError(
                            f"initial_conditions length {arr.size} != ndofs {ndofs}"
                        )
                u0_override = arr

            # Always compute and return a full TimeSeries (save_timeseries=True)
            solve_kwargs = dict(save_timeseries=True, outpath=out)
            if u0_override is not None:
                solve_kwargs["u0_override"] = u0_override
            if src is not None:
                solve_kwargs["source"] = src

            if steps is not None:
                solve_kwargs["steps"] = int(steps)
            u, U = solve(**solve_kwargs)

            if kwargs.get("visualize", False) and viz is not None:
                try:
                    viz(u)
                except Exception:
                    pass

            T = U.shape[0] if U is not None else 1
            t = np.arange(T) * float(meta.get("dt", 1.0))
            data = U if U is not None else u[np.newaxis, :]
            data = data.reshape(T, 1, -1, 1)  # (time, state, region, mode)
            labels_dimensions = {
                "State Variable": [str(meta.get("unknown", "u"))],
                "Region": [i for i in range(data.shape[2])],
            }
            ts = TimeSeries(
                time=t, data=data, network=None, labels_dimensions=labels_dimensions
            )
            return ts
        else:
            raise ValueError(
                f"Format {format} not supported. Valid formats: tvb, jax, python"
            )

        return simulation_data

    def get_experiment_file_prefix(self):
        atlas = (
            f"_atlas-{self.network.parcellation.atlas.name}"
            if self.network and self.network.parcellation
            else ""
        )
        return f"ses-{self.id}_desc-{self.local_dynamics.label}"

    @property
    def max_delay(self) -> float:
        """Compute the maximum delay (ms) from the current network/connectome."""
        return (
            float(np.max(self.network.compute_delays()))
            if self.network is not None
            else 0.0
        )

    @property
    def horizon(self, dt: float | None = None) -> int:
        """Number of history steps needed given delays and dt, like the old `horizon` attribute."""
        if dt is None:
            dt = float(self.integration.step_size)
        md = self.max_delay
        return int(round(md / dt)) + 1 if dt > 0 else 1

    def collect_initial_conditions(self, random=False):
        history = []
        n_modes = self.local_dynamics.metadata.number_of_modes
        n_nodes = self.network.number_of_regions

        if random:
            history.append(
                self.local_dynamics.get_initial_values(random=True, N=n_nodes)
            )
        else:
            for sv in self.local_dynamics.state_variables.values():
                history.append(np.repeat(sv.initial_value, n_nodes).astype(float))

        history = np.vstack(history)
        history = np.repeat(history[:, :, None], repeats=n_modes, axis=2)
        # Compute horizon from max delay and dt
        H = self.horizon
        md = self.max_delay
        history = np.repeat(history[None], repeats=H, axis=0)
        t = np.linspace(-md, 0, H)
        return TimeSeries(t, history)

    def save_model_specification(self, dir):
        file_prefix = self.get_experiment_file_prefix()
        lems_path = join(dir, f"{file_prefix}_simulation.xml")
        self.to_lems().export_to_file(lems_path)
        validate_lems(lems_path)
        return lems_path

    def to_lems(
        self,
        initial_conditions=1,
        out_path: str | None = None,
        out_file: str | None = None,
    ):
        import lems.api as lems
        from lems.model.component import Text
        from lems.model.simulation import DataWriter, Run

        model = self.local_dynamics.to_lems(initial_conditions=initial_conditions)

        base_local_ct = next(iter(model.component_types), None)
        local_comp = next(iter(model.components), None)

        local_ct = None
        if base_local_ct is not None:
            local_ct = lems.ComponentType(
                name="LocalDynamics", extends=base_local_ct.name
            )
            model.add(local_ct)
            if local_comp is not None:
                local_comp.type = local_ct.name

        sv_names = list(self.local_dynamics.state_variables.keys())
        target_sv = sv_names[0] if sv_names else "V"
        coupling_ct = lems.ComponentType(name="Coupling")
        coupling_ct.add(lems.Parameter(name="global_coupling", dimension="none"))

        try:
            coupl_meta = self.coupling
            params = getattr(coupl_meta, "parameters", {}) or {}
            for pname, pobj in params.items():
                pval = getattr(pobj, "value", 0)
                coupling_ct.add(
                    lems.Constant(name=str(pname), value=str(pval), dimension="none")
                )
            pre_expr = getattr(getattr(coupl_meta, "pre_expression", None), "rhs", None)
            post_expr = getattr(
                getattr(coupl_meta, "post_expression", None), "rhs", None
            )
        except Exception:
            params = {}
            pre_expr = f"{target_sv}_j"
            post_expr = "a*gx + b"

        import re as _re

        if isinstance(pre_expr, str):
            for m in _re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)_j\b", pre_expr):
                pname = m.group(1) + "_j"
                if all(
                    getattr(x, "name", None) != pname
                    for x in list(coupling_ct.parameters) + list(coupling_ct.constants)
                ):
                    coupling_ct.add(lems.Parameter(name=pname, dimension="none"))
        else:
            pre_expr = f"{target_sv}_j"

        coupling_ct.dynamics.add(
            lems.DerivedVariable(name="pre", value=str(pre_expr), dimension="none")
        )
        coupling_ct.dynamics.add(
            lems.DerivedVariable(
                name="gx", value="global_coupling * pre", dimension="none"
            )
        )
        coupling_ct.dynamics.add(
            lems.DerivedVariable(
                name="post",
                value=str(post_expr) if post_expr else "gx",
                dimension="none",
            )
        )
        coupling_ct.add(lems.DerivedParameter(name="c_pop0", value="post"))
        model.add(coupling_ct)

        comp_id = (
            local_comp.id
            if local_comp is not None
            else (local_ct.name if local_ct is not None else None)
        )
        if local_ct is not None and comp_id is not None:
            if "out_path" not in local_ct.texts:
                local_ct.add_text(Text("out_path"))
            if "out_file" not in local_ct.texts:
                local_ct.add_text(Text("out_file"))

            if local_comp is not None:
                dir_path = out_path or "."
                file_name = out_file
                if file_name is None:
                    base = os.path.basename(out_path) if out_path else ""
                    root, ext = os.path.splitext(base)
                    if base and ext:
                        dir_path = os.path.dirname(out_path) or "."
                        file_name = base
                    else:
                        file_name = base or "tvbo_timeseries.csv"
                        if "." not in file_name:
                            file_name = f"{file_name}.csv"
                local_comp.set_parameter("out_path", dir_path)
                local_comp.set_parameter("out_file", file_name)

            def ensure_ms(x):
                s = str(x).strip()
                return s if s.endswith("ms") else f"{s}ms"

            dt_ms = ensure_ms(self.integration.step_size)
            T_ms = ensure_ms(self.integration.duration)

            local_ct.simulation.add(Run(comp_id, "t", dt_ms, T_ms))
            local_ct.simulation.add(DataWriter("out_path", "out_file"))

            if comp_id not in model.targets:
                model.add_target(comp_id)

        for comp in model.components:
            for pk, pv in list(comp.parameters.items()):
                if not isinstance(pv, str):
                    comp.parameters[pk] = str(pv)

        return model

    def render_code(self, format="tvb", **kwargs):
        if format == "tvb":
            template = templates.lookup.get_template(
                "tvbo-tvb-SimulationExperiment.py.mako"
            )
            rendered_code = format_code(template.render(experiment=self))

        elif format.lower() in ["autodiff", "jax"]:
            template = templates.lookup.get_template("autodiff/tvbo-jax-sim.py.mako")
            rendered_code = format_code(
                template.render(experiment=self, **kwargs),
                use_black=False,
            )

        elif format in ["pde", "pde-fem", "pde-python"]:
            template = templates.lookup.get_template("tvbo-pde-fem.py.mako")
            rendered_code = format_code(
                template.render(experiment=self), use_black=True
            )

        return rendered_code

    def save_code(self, dir, file_name=None):
        if file_name is not None:
            file_prefix = self.get_experiment_file_prefix()
        else:
            file_prefix = file_name
        code_path = join(dir, f"{file_prefix}_simulation.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(self.render_code())

    def get_parameters_collection(self, **kwargs):
        if keys_to_exclude := kwargs.get("keys_to_exclude", []):
            keys_to_exclude = keys_to_exclude + ["connectivity", "coupling_terms"]
        parameters = Bunch()
        metadata.traverse_metadata(
            self,
            callback_kwargs={"parameters": parameters},
            keys_to_exclude=keys_to_exclude,
        )
        return parameters

    @property
    def parameters(self):
        return self.get_parameters_collection()

    # ---- Reporting utilities (paralleling Dynamics) ----
    def generate_report(
        self,
        format: str = "markdown",
        template_name: str = "tvbo-report-experiment",
        outputfile: str | None = None,
    ) -> str:
        """Render a human-readable report for this experiment.

        - Reuses the model/dynamics report template via Mako include to avoid redundancy.
        - Summarizes integration, network/connectome, coupling, monitors, stimulation, and software info.

        Parameters
        - format: 'markdown', 'html', or 'pdf' (pdf via pandoc)
        - template_name: base name of the template without extension
        - outputfile: optional path to write the rendered report
        """
        # Choose template
        if format in ["markdown", "md", "pdf"]:
            template = templates.lookup.get_template(f"report/{template_name}.md.mako")
        elif format in ["html", "htm"]:
            template = templates.lookup.get_template(
                f"report/{template_name}.html.mako"
            )
        else:
            raise ValueError("format must be one of: markdown, html, pdf")

        # Render with full experiment context; the template will include the model template
        render = template.render(experiment=self)

        # Persist if requested
        if outputfile:
            if format in ["pdf"]:
                from tvbo.export import report as _report

                _report.to_pdf(render, outputfile)
            else:
                with open(outputfile, "w", encoding="utf-8") as f:
                    f.write(render)

        return render

    def save_report(
        self, opath: str, format: str = "markdown", filename: str | None = None
    ):
        """Save the report to a file in the given directory.

        If filename is not provided, uses a sensible default based on experiment id and label.
        """
        os.makedirs(opath, exist_ok=True)
        if filename is None:
            base = f"experiment_{self.id}"
            if getattr(self, "label", None):
                base += f"_{self.label}"
            filename = base
        ext = (
            "md"
            if format in ["markdown", "md"]
            else ("html" if format in ["html", "htm"] else "pdf")
        )
        fpath = join(opath, f"{filename}.{ext}")
        self.generate_report(
            format=format if format != "md" else "markdown", outputfile=fpath
        )
