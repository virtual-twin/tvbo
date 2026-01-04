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
from tvbo.data.tvbo_data.connectomes import Network
from tvbo.data.types import SimulationState, TimeSeries
from tvbo.datamodel import tvbo_datamodel
from tvbo.export import templater
from tvbo.export.templater import format_code
from tvbo.knowledge import Coupling, Integrator
from tvbo.knowledge.simulation.localdynamics import Dynamics
from tvbo.knowledge.simulation.network import Coupling, _Network
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

        if "dynamics" in kwargs and not isinstance(kwargs["dynamics"], dict):
            # Convert list of Dynamics to dict keyed by name
            dynamics_input = kwargs["dynamics"]
            if isinstance(dynamics_input, list):
                dynamics_dict = {}
                for dyn in dynamics_input:
                    if dyn is not None and hasattr(dyn, "name"):
                        dynamics_dict[dyn.name] = dyn
                kwargs["dynamics"] = dynamics_dict
            elif hasattr(dynamics_input, "name"):
                # Single Dynamics instance - wrap in dict
                kwargs["dynamics"] = {dynamics_input.name: dynamics_input}

        # Ensure an id exists (the datamodel requires it in __post_init__)
        if kwargs.get("id") is None:
            kwargs["id"] = sessionid
            sessionid += 1

        # Handle pydantic Network - convert to dict before parent __init__
        # This prevents the parent's __post_init__ from failing on pydantic objects
        if "network" in kwargs and kwargs["network"] is not None:
            net = kwargs["network"]
            # Check if it's a pydantic model (has model_dump) - convert to dict
            if hasattr(net, "model_dump"):
                kwargs["network"] = net.model_dump(exclude_none=True)
            elif hasattr(net, "dict") and not isinstance(net, dict):
                # Pydantic v1
                kwargs["network"] = net.dict(exclude_none=True)

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

        if getattr(self, "network", None) and not isinstance(self.network, Network):
            self.network = _coerce(Network, self.network)

        # Mirror model/local_dynamics
        self.model = self.local_dynamics

        # If dynamics dict is empty, populate from local_dynamics
        if not getattr(self, "dynamics", None) and getattr(
            self, "local_dynamics", None
        ):
            ld = self.local_dynamics
            self.dynamics[ld.name] = ld

        # If local_dynamics is empty but dynamics dict exists, use first entry
        # This enables backwards-compatible single-model workflows
        if not getattr(self, "local_dynamics", None) and getattr(
            self, "dynamics", None
        ):
            dynamics_dict = self.dynamics
            if isinstance(dynamics_dict, dict) and dynamics_dict:
                first_key = next(iter(dynamics_dict))
                self.local_dynamics = dynamics_dict[first_key]
                self.model = self.local_dynamics

        if not getattr(self, "network", None):
            self.network = Network()

        # Get source file path if loading from file (set by from_file classmethod)
        self._source_file = getattr(self.__class__, '_pending_source_file', None)

        # Load network from BIDS if bids_dir is specified
        if hasattr(self.network, "bids_dir") and self.network.bids_dir:
            self._load_network_from_bids()

        if not getattr(self, "integration", None):
            self.integration = Integrator(method="Heun")

        if not getattr(self, "coupling", None):
            self.coupling = Coupling(name="Linear")

        if self.local_dynamics and not self.dynamics:
            self.dynamics[self.local_dynamics.name] = self.local_dyanmics

    def _load_network_from_bids(self):
        """Load network matrices from BEP017 BIDS directory.

        Uses network.bids_dir, network.structural_measures, and
        network.observational_measures to load connectivity data.
        Relative paths are resolved relative to the YAML source file.
        """
        from pathlib import Path

        bids_dir = Path(self.network.bids_dir)
        if not bids_dir.is_absolute():
            # Resolve relative to YAML source file location
            source_file = getattr(self, '_source_file', None)
            if source_file:
                bids_dir = (Path(source_file).parent / bids_dir).resolve()
            else:
                # Fallback: resolve relative to cwd
                bids_dir = (Path.cwd() / bids_dir).resolve()

        # Get measures from network attributes
        structural = getattr(self.network, "structural_measures", None) or [
            "streamlineCount", "tractLength"
        ]
        observational = getattr(self.network, "observational_measures", None) or []

        # Use Network.load_from_bids to load data into self.network
        self.network.load_from_bids(
            bids_dir,
            structural_measures=structural,
            observational_measures=observational,
        )

    @classmethod
    def from_datamodel(
        cls, dm: tvbo_datamodel.SimulationExperiment
    ) -> "SimulationExperiment":
        # Leverage the unified initializer
        return cls(**dm._as_dict)

    @classmethod
    def from_pyrates(cls, filepath: str) -> "SimulationExperiment":
        """Load a SimulationExperiment from a PyRates YAML template file.

        Parses all OperatorTemplates in the file and creates a keyed dict
        of Dynamics objects.

        Parameters
        ----------
        filepath : str
            Path to PyRates YAML file.

        Returns
        -------
        SimulationExperiment
            New instance with dynamics dict populated from PyRates templates.

        Example
        -------
        >>> exp = SimulationExperiment.from_pyrates("synaptic_plasticity.yaml")
        >>> print(exp.dynamics.keys())  # ['tsodyks', 'depression', 'facilitation']
        """
        from tvbo.export.pyrates import from_pyrates_yaml_all

        dynamics_dicts = from_pyrates_yaml_all(filepath)

        # Convert dicts to Dynamics instances using lightweight construction
        dynamics = {}
        for name, dyn_dict in dynamics_dicts.items():
            # Create instance with _skip_ontology=True to avoid slow lookups
            dyn = Dynamics(_skip_ontology=True, **dyn_dict)
            dynamics[name] = dyn

        # Use the first dynamics as local_dynamics if available
        local_dynamics = None
        if dynamics:
            first_key = next(iter(dynamics))
            local_dynamics = dynamics[first_key]

        return cls(
            dynamics=dynamics,
            local_dynamics=local_dynamics,
        )

    @classmethod
    def from_pydantic(cls, pyd_obj) -> "SimulationExperiment":
        """Create a SimulationExperiment from a Pydantic model instance.

        Args:
            pyd_obj: A Pydantic BaseModel instance (e.g., from tvbo.datamodel.tvbopydantic)

        Returns:
            SimulationExperiment instance
        """
        if hasattr(pyd_obj, "model_dump"):
            # Pydantic v2
            return cls(**pyd_obj.model_dump(exclude_none=True))
        elif hasattr(pyd_obj, "dict"):
            # Pydantic v1
            return cls(**pyd_obj.dict(exclude_none=True))
        else:
            raise TypeError(f"Expected a Pydantic model, got {type(pyd_obj)}")

    @classmethod
    def from_tvb_simulator(cls, tvb_simulator):
        return cls.from_datamodel(metadata.simulator2metadata(tvb_simulator))

    @classmethod
    def from_file(cls, filepath: str):
        from linkml_runtime.loaders import yaml_loader
        from pathlib import Path

        # Store source file path BEFORE loading so __init__ can use it
        cls._pending_source_file = str(Path(filepath).resolve())
        try:
            exp = yaml_loader.load(filepath, target_class=cls)
            exp._source_file = cls._pending_source_file
        finally:
            cls._pending_source_file = None
        return exp

    @classmethod
    def from_string(cls, yaml_string: str) -> "SimulationExperiment":
        """Create a SimulationExperiment from a YAML string.

        This is useful for defining experiments inline in notebooks or scripts
        using human-readable YAML syntax.

        Parameters
        ----------
        yaml_string : str
            YAML-formatted string defining the experiment.

        Returns
        -------
        SimulationExperiment
            New instance populated from the YAML definition.

        Example
        -------
        >>> exp = SimulationExperiment.from_string('''
        ... id: 1
        ... label: My Experiment
        ... local_dynamics:
        ...   name: JansenRit
        ...   parameters:
        ...     A: {value: 3.25}
        ... ''')
        """
        from linkml_runtime.loaders import yaml_loader

        return yaml_loader.loads(yaml_string, target_class=cls)

    @classmethod
    def from_bids(
        cls,
        bids_dir: str,
        subject: str = "01",
        session: str | None = None,
        run_to_verify: bool = False,
    ) -> tuple["SimulationExperiment", TimeSeries]:
        """
        Load a SimulationExperiment and TimeSeries from a BIDS BEP034 dataset.

        This method ingests data exported via `to_bids()` and reconstructs:
        - The SimulationExperiment with model, network, and integration settings
        - The TimeSeries data (with 100% fidelity for HDF5 format)

        Automatically detects the time series format (HDF5, CIFTI, or TSV).

        Parameters
        ----------
        bids_dir : str
            Path to the root BIDS dataset directory (e.g., './derivatives/tvbo')
        subject : str
            Subject identifier (with or without 'sub-' prefix). Default: '01'
        session : str, optional
            Session identifier (with or without 'ses-' prefix).
            If not specified and sessions exist, uses the first one.
        run_to_verify : bool
            If True, re-run the simulation and compare with loaded TimeSeries.
            Useful for verifying reproducibility. Default: False.

        Returns
        -------
        tuple[SimulationExperiment, TimeSeries]
            The reconstructed experiment and time series data.

        Examples
        --------
        >>> # Load from BIDS
        >>> exp, ts = SimulationExperiment.from_bids("./derivatives/tvbo", subject="01")
        >>> print(ts.shape)
        (1000, 2, 68, 1)

        >>> # Verify reproducibility
        >>> exp, ts = SimulationExperiment.from_bids(
        ...     "./derivatives/tvbo",
        ...     subject="01",
        ...     run_to_verify=True
        ... )

        >>> # Access the experiment settings
        >>> print(exp.local_dynamics.name)
        'Generic2dOscillator'

        Notes
        -----
        - HDF5 format preserves full dimensionality with 100% fidelity
        - CIFTI/TSV formats reconstruct from per-state-variable files
        - Model parameters are restored from eq/ sidecar if available
        - Network connectivity is restored from net/ directory

        See Also
        --------
        to_bids : Export experiment to BIDS format
        """
        from pathlib import Path

        from tvbo.export.bids import (
            ingest_bids_session,
        )

        # Ingest all data from the BIDS session
        bids_data = ingest_bids_session(bids_dir, subject, session)

        # =====================================================================
        # 1. Reconstruct Network
        # =====================================================================
        network = None
        if bids_data["network"] is not None:
            net_data = bids_data["network"]
            network = Network(
                weights=net_data["weights"],
                tract_lengths=net_data.get("distances"),
                region_labels=(
                    np.array(net_data["region_labels"])
                    if net_data["region_labels"]
                    else None
                ),
            )

            # Add coordinates if available
            if bids_data["coordinates"] is not None:
                coord_data = bids_data["coordinates"]
                if coord_data["centres"] is not None:
                    network.centres = coord_data["centres"]

        # =====================================================================
        # 2. Reconstruct Dynamics model
        # =====================================================================
        local_dynamics = None
        if bids_data["equations"] is not None:
            eq_data = bids_data["equations"]
            model_type = eq_data.get("model_type", "Generic2dOscillator")
            params = eq_data.get("parameters", {})

            # Try to load from ontology by name
            try:
                local_dynamics = Dynamics.from_ontology(model_type)
                # Apply stored parameters
                for param_name, param_value in params.items():
                    if (
                        hasattr(local_dynamics, "parameters")
                        and param_name in local_dynamics.parameters
                    ):
                        local_dynamics.parameters[param_name].value = param_value
            except Exception:
                # Fallback: create minimal Dynamics
                local_dynamics = Dynamics(name=model_type)
        else:
            # Default dynamics
            local_dynamics = Dynamics.from_ontology("Generic2dOscillator")

        # =====================================================================
        # 3. Reconstruct Integration settings from sidecar provenance
        # =====================================================================
        integration = Integrator(method="Heun")

        if bids_data["timeseries"] is not None:
            ts_data = bids_data["timeseries"]

            # Get sampling from time series
            if ts_data["sample_period"] is not None:
                sample_period = ts_data["sample_period"]
                # Convert to step size (usually smaller)
                # Note: step_size may not equal sample_period if subsampling was used
                integration.step_size = sample_period

            # Check sidecars for provenance info
            for sidecar in ts_data.get("sidecars", []):
                provenance = sidecar.get("Provenance") or sidecar.get(
                    "SimulationProvenance"
                )
                if provenance:
                    if "StepSize" in provenance and provenance["StepSize"]:
                        integration.step_size = float(provenance["StepSize"])
                    if "Duration" in provenance and provenance["Duration"]:
                        integration.duration = float(provenance["Duration"])
                    if "Integrator" in provenance and provenance["Integrator"]:
                        # Try to parse integrator method from string
                        int_str = str(provenance["Integrator"])
                        for method in ["Heun", "Euler", "RungeKutta"]:
                            if method.lower() in int_str.lower():
                                integration.method = method
                                break

        # =====================================================================
        # 4. Reconstruct TimeSeries
        # =====================================================================
        timeseries = None
        if bids_data["timeseries"] is not None:
            ts_data = bids_data["timeseries"]

            # Build labels_dimensions
            labels_dimensions = ts_data.get("labels_dimensions", {})
            if not labels_dimensions:
                labels_dimensions = {
                    "State Variable": ts_data["state_variables"],
                    "Space": ts_data["region_labels"],
                }

            timeseries = TimeSeries(
                time=ts_data["time"],
                data=ts_data["data"],
                network=network,
                title=f"Loaded from BIDS: sub-{subject}",
                sample_period=ts_data["sample_period"],
                labels_dimensions=labels_dimensions,
            )
            timeseries.sample_period_unit = ts_data.get("sample_period_unit", "ms")

            # Store source info
            timeseries._bids_source = {
                "bids_dir": str(bids_dir),
                "subject": subject,
                "session": session,
                "format": ts_data["format"],
            }

        # =====================================================================
        # 5. Create SimulationExperiment
        # =====================================================================
        experiment = cls(
            label=f"Loaded from BIDS: sub-{subject}",
            description=f"Experiment reconstructed from BIDS dataset at {bids_dir}",
            local_dynamics=local_dynamics,
            network=network,
            integration=integration,
            coupling=Coupling(name="Linear"),  # Default coupling
        )

        # Store BIDS source info
        experiment._bids_source = {
            "bids_dir": str(bids_dir),
            "subject": subject,
            "session": session,
            "loaded_at": str(np.datetime64("now")),
        }

        # =====================================================================
        # 6. Verify reproducibility if requested
        # =====================================================================
        if run_to_verify and timeseries is not None:
            print("Running simulation to verify reproducibility...")
            ts_rerun = experiment.run(format="jax")

            # Compare shapes
            if ts_rerun.shape != timeseries.shape:
                print(
                    f"WARNING: Shape mismatch! Loaded: {timeseries.shape}, Rerun: {ts_rerun.shape}"
                )
            else:
                # Compare data
                max_diff = np.max(
                    np.abs(np.asarray(ts_rerun.data) - np.asarray(timeseries.data))
                )
                if max_diff < 1e-6:
                    print(f"✓ Verification passed! Max difference: {max_diff:.2e}")
                else:
                    print(f"⚠ Data differs. Max difference: {max_diff:.2e}")
                    print(
                        "  This may be expected if noise was used or parameters differ."
                    )

        return experiment, timeseries

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
        import dataclasses

        cls = self.__class__
        # For dataclasses, we need to copy all fields, not just __dict__
        # __dict__ may not include fields that are still at their default values
        data = {}
        if dataclasses.is_dataclass(self):
            for field in dataclasses.fields(self):
                value = getattr(self, field.name, None)
                data[field.name] = _copy.deepcopy(value, memo)
        else:
            # Fallback for non-dataclass
            for k, v in self.__dict__.items():
                data[k] = _copy.deepcopy(v, memo)

        # Create clone using proper constructor to ensure all defaults are set
        clone = cls(**data)
        memo[id(self)] = clone
        return clone

    def to_yaml(self, filepath: str | None = None, format: str = "tvbo") -> str:
        """Export the experiment to YAML format.

        Parameters
        ----------
        filepath : str, optional
            Path to write the YAML file. If None, returns the YAML string.
        format : str
            Output format: "tvbo" (default) or "pyrates".

        Returns
        -------
        str
            YAML string or filepath if written to file.
        """
        if format.lower() == "pyrates":
            from tvbo.export.pyrates import to_pyrates_yaml_string
            from tvbo.knowledge.simulation import Dynamics as DynamicsClass

            # Get network
            network = getattr(self, "network", None)

            # Handle dynamics based on whether we have a network or single model
            if network is not None:
                # Network case: pass dynamics as dict for heterogeneous networks
                dynamics = self.dynamics
                if dynamics is None:
                    dynamics = {}
                elif not isinstance(dynamics, dict):
                    # Convert list to dict keyed by name
                    if isinstance(dynamics, list):
                        dynamics = {d.name: d for d in dynamics if d is not None}
                    else:
                        # Single model - wrap in dict
                        dynamics = {dynamics.name: dynamics} if dynamics else {}

                # Convert all datamodel Dynamics to full Dynamics class with methods
                dynamics_converted = {}
                for name, dyn in dynamics.items():
                    if dyn is not None and not hasattr(dyn, "render_equation"):
                        dynamics_converted[name] = DynamicsClass.from_datamodel(dyn)
                    else:
                        dynamics_converted[name] = dyn

                return to_pyrates_yaml_string(
                    dynamics=dynamics_converted,
                    network=network,
                    filepath=filepath,
                )
            else:
                # Single model case (no network)
                dynamics = getattr(self, "local_dynamics", None)
                if dynamics is None:
                    dynamics = self.dynamics
                    if isinstance(dynamics, list) and len(dynamics) == 1:
                        dynamics = dynamics[0]
                    elif isinstance(dynamics, dict) and len(dynamics) == 1:
                        dynamics = list(dynamics.values())[0]

                # Convert datamodel Dynamics to full Dynamics class with methods
                if dynamics is not None and not hasattr(dynamics, "render_equation"):
                    dynamics = DynamicsClass.from_datamodel(dynamics)

                return to_pyrates_yaml_string(
                    dynamics=dynamics,
                    network=network,
                    filepath=filepath,
                )
        else:
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
                    meta_list.append(tvbo_datamodel.Observation(**m))
        if meta_list:
            self.monitors = meta_list

    def configure(self):
        # Disable delayed logic if the connectome has no path lengths or conduction speed is infinite
        try:
            network = getattr(self, "network", None)
            if network is None:
                return

            # Get the network as a Network (it might already be one)
            if isinstance(network, Network):
                conn = network
            else:
                conn = Network(network)

            # Try to get lengths matrix
            try:
                L = conn.lengths_matrix
            except Exception:
                L = None

            # Disable delays if lengths are None or all zeros
            if (
                L is None
                or np.allclose(L, 0)
                or np.allclose(L.max() / conn.conduction_speed.value, 0)
            ):
                if getattr(self, "integration", None) is not None:
                    self.integration.delayed = False
                if getattr(self, "coupling", None) is not None:
                    self.coupling.delayed = False
        except Exception as e:
            # Best-effort; keep defaults if anything goes wrong
            import warnings

            warnings.warn(f"Could not configure delays: {e}")

    def add_stimulus(self, stimulus):
        import owlready2 as owl

        from tvbo.knowledge.simulation import perturbation

        if isinstance(stimulus, perturbation.Stimulus):
            self.stimulation = stimulus
        elif isinstance(stimulus, str) or isinstance(stimulus, owl.ThingClass):
            self.stimulation = perturbation.Stimulus.from_ontology(stimulus)

    def collect_state(self, initial_conditions: TimeSeries | None = None):
        _ = self.noise_sigma_array
        parameters = self.get_parameters_collection(
            keys_to_exclude=[
                "derived_parameters",
                "conduction_speed",
                "coupling_terms",
            ]
        )
        # Expand coupling parameters with shape annotations like "(N, N)" or "(N,)"
        self._expand_coupling_parameter_shapes(parameters)

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
            parameters=parameters,
            stimulus=None,
            monitor_parameters=None,
        )
        # Attach state variable names for ergonomic noise setters
        try:
            state._svar_names = list(self.local_dynamics.state_variables.keys())
        except Exception:
            pass
        return state

    def _expand_coupling_parameter_shapes(self, parameters: Bunch) -> None:
        """Expand coupling parameters that have shape annotations like (N, N) or (N,)."""
        if not hasattr(parameters, "coupling") or self.network is None:
            return

        N = self.network.number_of_nodes
        coupling_params = parameters.coupling

        # Get the coupling parameter definitions with shape info
        if self.coupling is None:
            return

        for param_name, param_obj in (self.coupling.parameters or {}).items():
            if param_name not in coupling_params:
                continue

            shape_str = getattr(param_obj, "shape", None)
            if not shape_str:
                continue

            current_value = coupling_params[param_name]

            # Parse shape string and expand
            if shape_str == "(N, N)" or shape_str == "(N,N)":
                # Expand scalar to NxN matrix
                if np.isscalar(current_value) or (
                    hasattr(current_value, "shape") and current_value.shape == ()
                ):
                    coupling_params[param_name] = np.full((N, N), float(current_value))
            elif shape_str == "(N,)" or shape_str == "(N)":
                # Expand scalar to N-vector
                if np.isscalar(current_value) or (
                    hasattr(current_value, "shape") and current_value.shape == ()
                ):
                    coupling_params[param_name] = np.full((N,), float(current_value))

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

        elif format.lower() in ["tvboptim", "tvb-optim"]:
            # Return namespace with all generated functions for tvboptim workflows
            # This allows: ns = exp.execute('tvboptim')
            #              spectrum, cauchy_pdf = ns.spectrum, ns.cauchy_pdf
            namespace = {}
            code = self.render_code("tvboptim")
            exec(code, namespace)
            return Bunch(**namespace)

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
            raise ValueError(
                f"Format {format} not supported. Valid formats: tvb, tvboptim, jax."
            )

    def run(self, format="jax", initial_conditions=None, **kwargs):
        if "duration" in kwargs:
            self.integration.duration = kwargs.pop("duration")

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
            # Link TimeSeries to source experiment for provenance tracking
            ts.source_experiment = self
            return ts

        elif format.lower() in ["tvboptim", "tvb-optim"]:
            import time

            benchmark = kwargs.pop("benchmark", False)
            timings = Bunch() if benchmark else None

            # Get the namespace (reuse execute to avoid code duplication)
            if benchmark:
                t0 = time.perf_counter()
            ns = self.execute("tvboptim")
            if benchmark:
                timings.code_generation = time.perf_counter() - t0

            # Mode defaults to 'all' - run complete workflow
            mode = kwargs.pop("mode", "all")

            # Run the experiment with optional per-step timing
            if benchmark:
                # Run with detailed timing
                t0 = time.perf_counter()
                results = ns.run_experiment(
                    weights=self.network.weights,
                    distances=self.network.distances,
                    mode=mode,
                    **kwargs,
                )
                # Wait for JAX async dispatch to complete
                jax.block_until_ready(results.result.data if hasattr(results, 'result') else results)
                timings.total = time.perf_counter() - t0

                # Add timings to results
                results.timings = timings
                return results
            else:
                return ns.run_experiment(
                    weights=self.network.weights,
                    distances=self.network.distances,
                    mode=mode,
                    **kwargs,
                )

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

            # Link TimeSeries to source experiment for provenance tracking
            ts.source_experiment = self

            return ts

        elif format.lower() == "python":
            bnm = _Network(Network(self.network))
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
            # Link TimeSeries to source experiment for provenance tracking
            ts.source_experiment = self
            return ts

        elif format.lower() == "pyrates":
            return self._run_pyrates(**kwargs)

        else:
            raise ValueError(
                f"Format {format} not supported. Valid formats: tvb, jax, python, pyrates"
            )

        return simulation_data

    def _run_pyrates(
        self,
        solver: str | None = None,
        inputs: dict | None = None,
        outputs: list[str] | None = None,
        matrix_edge_threshold: int = 100,
        **kwargs,
    ) -> TimeSeries:
        """Run simulation using PyRates backend.

        Parameters
        ----------
        solver : str, optional
            ODE solver: "euler", "heun", "scipy". Defaults to mapped integration.method.
        inputs : dict, optional
            External inputs as {node/op/var: array} or will be auto-generated.
        outputs : list[str], optional
            Variables to monitor. If None, monitors all state variables.
        matrix_edge_threshold : int, optional
            For networks with N > threshold nodes, use add_edges_from_matrix instead
            of YAML edges for efficiency. Default is 100.
        **kwargs
            Additional kwargs passed to circuit.run().

        Returns
        -------
        TimeSeries
            TVBO TimeSeries with simulation results.
        """
        import shutil
        import sys
        import tempfile
        import uuid
        from pyrates.frontend import CircuitTemplate
        from pyrates import clear

        # Get simulation parameters from metadata
        integration = getattr(self, "integration", None)

        # PyRates supported solvers
        PYRATES_SOLVERS = {"euler", "heun", "scipy"}
        TVBO_TO_PYRATES_SOLVER = {
            "EulerDeterministic": "euler",
            "Euler": "euler",
            "HeunDeterministic": "heun",
            "Heun": "heun",
            "RungeKutta4thOrder": "scipy",
            "RungeKutta4": "scipy",
            "RK4": "scipy",
        }

        if solver is None:
            method = getattr(integration, "method", None)
            if method in TVBO_TO_PYRATES_SOLVER:
                solver = TVBO_TO_PYRATES_SOLVER[method]
            elif method in PYRATES_SOLVERS:
                solver = method
            elif method:
                raise ValueError(
                    f"Unsupported integration method '{method}' for PyRates. "
                    f"Supported: {list(TVBO_TO_PYRATES_SOLVER.keys())}"
                )
            else:
                solver = "heun"
        elif solver not in PYRATES_SOLVERS:
            raise ValueError(
                f"Invalid solver '{solver}'. Supported: {sorted(PYRATES_SOLVERS)}"
            )

        network = getattr(self, "network", None)
        n_nodes = 0
        if network is not None:
            n_nodes = getattr(network, "number_of_nodes", 0) or (
                len(network.nodes) if hasattr(network, "nodes") and network.nodes else 0
            )

        # For large networks, use matrix-based edges (much faster)
        use_matrix_edges = n_nodes > matrix_edge_threshold and hasattr(
            network, "weights_matrix"
        )

        # Build circuit from YAML (operators, nodes, and optionally edges)
        circuit, tmpdir, pkg_name = self._load_pyrates_circuit_from_yaml(
            include_edges=not use_matrix_edges
        )

        try:
            # For large networks, add edges via matrix
            if use_matrix_edges:
                self._add_pyrates_edges_from_matrix(circuit, network)

            # Build outputs/inputs if not provided
            if outputs is None:
                outputs = self._build_pyrates_outputs()
            if inputs is None:
                inputs = self._build_pyrates_inputs()

            # Run simulation
            result = circuit.run(
                step_size=self.integration.step_size,
                simulation_time=self.integration.duration,
                inputs=inputs,
                outputs=outputs,
                solver=solver,
                **kwargs,
            )
            clear(circuit)

        except Exception:
            clear(circuit)
            raise
        finally:
            # Cleanup
            if tmpdir in sys.path:
                sys.path.remove(tmpdir)
            modules_to_remove = [k for k in sys.modules if k.startswith(pkg_name)]
            for mod in modules_to_remove:
                del sys.modules[mod]
            shutil.rmtree(tmpdir, ignore_errors=True)

        # Compute algebraic output variables post-hoc
        result = self._compute_pyrates_outputs(result)

        return self._pyrates_result_to_timeseries(result)

    def _load_pyrates_circuit_from_yaml(self, include_edges: bool = True) -> tuple:
        """Load PyRates circuit from YAML template.

        Parameters
        ----------
        include_edges : bool
            If True, include edges in YAML. If False, edges will be added separately.

        Returns
        -------
        tuple
            (circuit, tmpdir, pkg_name) for cleanup
        """
        import sys
        import tempfile
        import uuid
        from pyrates.frontend import CircuitTemplate

        # Export to PyRates YAML
        yaml_content = self.to_yaml(format="pyrates")

        # For large networks, strip edges from YAML (we'll add them via matrix)
        if not include_edges:
            yaml_content = self._strip_edges_from_yaml(yaml_content)

        # Create temporary package
        tmpdir = tempfile.mkdtemp(prefix="tvbo_pyrates_")
        pkg_name = f"_tvbo_pyrates_{uuid.uuid4().hex[:8]}"
        pkg_path = os.path.join(tmpdir, pkg_name)
        os.makedirs(pkg_path, exist_ok=True)
        open(os.path.join(pkg_path, "__init__.py"), "w").close()

        yaml_path = os.path.join(pkg_path, "model.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        sys.path.insert(0, tmpdir)

        # Get circuit name
        network = getattr(self, "network", None)
        dynamics = getattr(self, "local_dynamics", None) or getattr(
            self, "dynamics", None
        )

        if network is not None:
            circuit_name = (
                getattr(network, "label", None)
                or getattr(network, "name", None)
                or "tvbo_circuit"
            )
        elif dynamics is not None:
            model_name = getattr(dynamics, "name", None) or "tvbo_model"
            circuit_name = f"{model_name}_circuit"
        else:
            circuit_name = "tvbo_circuit"

        circuit = CircuitTemplate.from_yaml(f"{pkg_name}.model.{circuit_name}")
        return circuit, tmpdir, pkg_name

    def _strip_edges_from_yaml(self, yaml_content: str) -> str:
        """Remove edges section from YAML for large networks."""
        import re

        # Remove edges section from circuit definition
        # Match "  edges:\n" followed by lines starting with "    -"
        pattern = r"(  edges:\n(?:    - \[.*\]\n)*)"
        return re.sub(pattern, "", yaml_content)

    def _add_pyrates_edges_from_matrix(self, circuit, network) -> None:
        """Add edges to circuit using weight matrix (efficient for large networks)."""
        dynamics_dict = self.dynamics
        if not isinstance(dynamics_dict, dict):
            dynamics_dict = {d.name: d for d in (dynamics_dict or [])}

        # Get node labels
        node_labels = []
        if hasattr(network, "nodes") and network.nodes:
            for node in network.nodes:
                label = getattr(node, "label", None) or f"node_{node.id}"
                node_labels.append(str(label).replace(" ", "_").replace("-", "_"))
        else:
            n_nodes = network.weights_matrix.shape[0]
            node_labels = [f"node_{i}" for i in range(n_nodes)]

        # Get source/target variables from dynamics
        first_dyn = next(iter(dynamics_dict.values()))
        dyn_name = first_dyn.name

        # Source: prefer output, fallback to first state variable
        if first_dyn.output:
            src_var = f"{dyn_name}_op/{list(first_dyn.output.keys())[0]}"
        elif first_dyn.state_variables:
            src_var = f"{dyn_name}_op/{list(first_dyn.state_variables.keys())[0]}"
        else:
            src_var = f"{dyn_name}_op/x"

        # Target: coupling term
        if first_dyn.coupling_terms:
            tgt_var = f"{dyn_name}_op/{list(first_dyn.coupling_terms.keys())[0]}"
        else:
            tgt_var = src_var

        # Add edges using matrix (MUCH faster!)
        weights = network.weights_matrix
        if weights is not None and weights.size > 0:
            # Get delays - priority: explicit delays > computed from distances/lengths
            delays = None

            # First check for explicit delays from edges
            if hasattr(network, "_delays_from_edges"):
                delays = network._delays_from_edges()

            # If no explicit delays, compute from lengths/distances
            if delays is None:
                lengths = getattr(network, "lengths_matrix", None)
                if lengths is not None and np.any(lengths > 0):
                    delays = network.calculate_delays()

            # Build edge_attr dict if we have delays
            edge_attr = None
            if delays is not None:
                edge_attr = {"delay": delays}

            circuit.add_edges_from_matrix(
                source_var=src_var,
                target_var=tgt_var,
                source_nodes=node_labels,
                weight=weights,
                edge_attr=edge_attr,
                min_weight=1e-12,
            )

    def _build_pyrates_outputs(self) -> dict:
        """Build PyRates outputs dict from dynamics state variables.

        Note: PyRates only tracks state variables (differential equations) in its
        state vector. Algebraic outputs like 'v_pyr = y1 - y2' are computed inline
        but not stored for recording. We only request state variables here, and
        compute outputs post-hoc in _run_pyrates using _compute_pyrates_outputs.
        """
        outputs = {}

        # Get dynamics - prefer dict form
        dynamics = self.dynamics
        if not isinstance(dynamics, dict):
            dynamics = {d.name: d for d in (dynamics or [])}

        # Get default dynamics (first in dict)
        default_dyn = next(iter(dynamics.values())) if dynamics else None

        def add_outputs_for_dynamics(dyn, op_name: str, prefix: str = ""):
            """Add state variables for a dynamics model."""
            if not dyn:
                return
            # Add state variables only (PyRates can only record these)
            for sv_name in (dyn.state_variables or {}).keys():
                key = f"{prefix}{sv_name}" if prefix else sv_name
                outputs[key] = f"{prefix.rstrip('_') or 'node_0'}/{op_name}/{sv_name}"

        # Get network nodes if available
        network = getattr(self, "network", None)
        if network is not None and hasattr(network, "nodes") and network.nodes:
            for node in network.nodes:
                node_label = getattr(node, "label", None) or f"node_{node.id}"
                safe_label = str(node_label).replace(" ", "_").replace("-", "_")

                # Get dynamics for this node, fall back to default
                dyn_name = (
                    node.dynamics
                    if isinstance(node.dynamics, str)
                    else getattr(node.dynamics, "name", None)
                )
                dyn = dynamics.get(dyn_name) if dyn_name else default_dyn

                if dyn:
                    op_name = f"{dyn.name}_op"
                    add_outputs_for_dynamics(dyn, op_name, prefix=f"{safe_label}_")
        else:
            # Single dynamics case
            if dynamics:
                dyn = next(iter(dynamics.values()))
                op_name = f"{dyn.name}_op"
                add_outputs_for_dynamics(dyn, op_name)

        return outputs

    def _compute_pyrates_outputs(self, result: "pd.DataFrame") -> "pd.DataFrame":
        """Compute algebraic output variables from PyRates simulation results.

        PyRates only records state variables. This method evaluates output
        equations (like 'v_pyr = y1 - y2') using the recorded state values.
        """
        import sympy as sp

        # Get dynamics
        dynamics = self.dynamics
        if not isinstance(dynamics, dict):
            dynamics = {d.name: d for d in (dynamics or [])}

        if not dynamics:
            return result

        # Get default dynamics
        default_dyn = next(iter(dynamics.values()))

        # Compute outputs for each dynamics model
        network = getattr(self, "network", None)
        if network is not None and hasattr(network, "nodes") and network.nodes:
            for node in network.nodes:
                node_label = getattr(node, "label", None) or f"node_{node.id}"
                safe_label = str(node_label).replace(" ", "_").replace("-", "_")
                prefix = f"{safe_label}_"

                # Get dynamics for this node
                dyn_name = (
                    node.dynamics
                    if isinstance(node.dynamics, str)
                    else getattr(node.dynamics, "name", None)
                )
                dyn = dynamics.get(dyn_name) if dyn_name else default_dyn

                if dyn and dyn.output:
                    for out_name, out_var in dyn.output.items():
                        eq = out_var.equation
                        if eq and eq.rhs:
                            # Parse and evaluate the equation
                            expr = sp.sympify(eq.rhs)
                            # Substitute state variable values
                            subs = {}
                            for sym in expr.free_symbols:
                                col_name = f"{prefix}{sym.name}"
                                if col_name in result.columns:
                                    subs[sym] = result[col_name].values
                            if subs:
                                # Add output with same prefix as state variables
                                out_col = f"{prefix}{out_name}"
                                # Vectorized evaluation using lambdify
                                func = sp.lambdify(list(subs.keys()), expr, "numpy")
                                result[out_col] = func(*subs.values())
        else:
            # Single dynamics case (no network/nodes)
            dyn = default_dyn
            if dyn and dyn.output:
                for out_name, out_var in dyn.output.items():
                    eq = out_var.equation
                    if eq and eq.rhs:
                        expr = sp.sympify(eq.rhs)
                        subs = {}
                        for sym in expr.free_symbols:
                            if sym.name in result.columns:
                                subs[sym] = result[sym.name].values
                        if subs:
                            func = sp.lambdify(list(subs.keys()), expr, "numpy")
                            result[out_name] = func(*subs.values())

        return result

    def _build_pyrates_inputs(self) -> dict:
        """Build PyRates inputs dict from experiment stimulation.

        Converts TVBO Stimulus objects to PyRates input format:
        {"NodeLabel/DynamicsName_op/I_ext": np.array([...])}

        Returns
        -------
        dict
            PyRates-compatible inputs dictionary.
        """
        inputs = {}

        stimulation = getattr(self, "stimulation", None)
        if stimulation is None:
            return inputs

        # Get time array for stimulus evaluation
        duration = self.integration.duration
        step_size = self.integration.step_size
        time = np.arange(0, duration, step_size)

        # Get the stimulus function
        try:
            stim_func = stimulation.execute(format="python")
        except Exception:
            # If execute fails, try to evaluate directly
            stim_func = None

        if stim_func is None:
            return inputs

        # Evaluate stimulus over time
        stim_values = stim_func(time)

        # Determine target variable (default to I_ext for rate neurons)
        target_var = getattr(stimulation, "target_variable", None) or "I_ext"

        # Get regions/nodes to stimulate
        regions = getattr(stimulation, "regions", None) or []
        weighting = getattr(stimulation, "weighting", None) or []

        # Get network nodes
        network = getattr(self, "network", None)
        dynamics = self.dynamics
        if not isinstance(dynamics, dict):
            dynamics = {d.name: d for d in (dynamics or [])}

        if network is not None and hasattr(network, "nodes") and network.nodes:
            nodes = list(network.nodes)

            # If no specific regions, apply to first node (Pre-synaptic)
            if not regions:
                regions = [0]

            for i, region_idx in enumerate(regions):
                if region_idx >= len(nodes):
                    continue

                node = nodes[region_idx]
                node_label = getattr(node, "label", None) or f"node_{node.id}"
                safe_label = str(node_label).replace(" ", "_").replace("-", "_")

                # Get dynamics name for this node
                dyn_name = (
                    node.dynamics
                    if isinstance(node.dynamics, str)
                    else getattr(node.dynamics, "name", None)
                )

                if dyn_name:
                    op_name = f"{dyn_name}_op"
                    key = f"{safe_label}/{op_name}/{target_var}"

                    # Apply weighting if available
                    weight = weighting[i] if i < len(weighting) else 1.0
                    inputs[key] = stim_values * weight

        return inputs

    def _pyrates_result_to_timeseries(self, result) -> TimeSeries:
        """Convert PyRates pandas DataFrame result to TVBO TimeSeries.

        Reshapes data from PyRates flat format to TVBO's standard format:
        (time, state_variables, nodes, modes)
        """
        time = np.array(result.index)
        columns = list(result.columns)

        # Get known node labels from network
        known_node_labels = []
        network = getattr(self, "network", None)
        if network is not None and hasattr(network, "nodes") and network.nodes:
            for node in network.nodes:
                node_label = getattr(node, "label", None) or f"node_{node.id}"
                safe_label = str(node_label).replace(" ", "_").replace("-", "_")
                known_node_labels.append(safe_label)

        # Parse column names to extract node and state variable info
        # Columns are named like "NodeLabel_statevariable" (e.g., "node_0_y1", "node_0_v_pyr")
        node_names = []
        sv_names = []
        node_sv_pairs = []

        for col in columns:
            # Try to match against known node labels first
            node_name = None
            sv_name = None
            for node_label in known_node_labels:
                prefix = f"{node_label}_"
                if col.startswith(prefix):
                    node_name = node_label
                    sv_name = col[len(prefix) :]
                    break

            if node_name is None:
                # Fallback: split on last underscore
                parts = col.rsplit("_", 1)
                if len(parts) == 2:
                    node_name, sv_name = parts
                else:
                    node_name, sv_name = col, col

            node_sv_pairs.append((node_name, sv_name))
            if node_name not in node_names:
                node_names.append(node_name)
            if sv_name not in sv_names:
                sv_names.append(sv_name)

        n_time = len(time)
        n_nodes = len(node_names)
        n_svs = len(sv_names)

        # Create properly shaped data array: (time, state_vars, nodes, modes=1)
        data = np.zeros((n_time, n_svs, n_nodes, 1), dtype=np.float32)

        # Fill in the data by mapping each column to its (sv_index, node_index)
        for col_idx, (node_name, sv_name) in enumerate(node_sv_pairs):
            node_idx = node_names.index(node_name)
            sv_idx = sv_names.index(sv_name)
            data[:, sv_idx, node_idx, 0] = result.iloc[:, col_idx].values

        labels_dimensions = {
            "State Variable": sv_names,
            "Region": node_names,
        }

        return TimeSeries(
            time=time,
            data=data,
            labels_dimensions=labels_dimensions,
            sample_period=float(time[1] - time[0]) if len(time) > 1 else 1.0,
        )

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
        if self.network is None:
            return 0.0
        delays = self.network.compute_delays()
        # Use nanmax to ignore NaN values for non-existent edges
        max_val = np.nanmax(delays)
        return float(max_val) if not np.isnan(max_val) else 0.0

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

        elif format.lower() == "tvboptim":
            template = templates.lookup.get_template(
                "tvboptim/tvbo-tvboptim-experiment.py.mako"
            )
            rendered_code = format_code(
                template.render(experiment=self, **kwargs),
                use_black=False,
            )

        else:
            raise ValueError(
                f"Unknown format: {format}. Supported: tvb, autodiff, jax, pde, tvboptim"
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

    def to_bids(
        self,
        output_dir: str,
        subject: str = "01",
        session: str | None = None,
        description: str = "tvbsim",
        run: int | None = None,
        ts_label: str = "sim",
        timeseries: TimeSeries | None = None,
        run_simulation: bool = True,
        **run_kwargs,
    ) -> str:
        """
        Export simulation experiment and results to BIDS-compliant format (BEP034 v1.0.0).

        This method creates a complete BIDS dataset containing:
        - Time series data in ts/ directory
        - Network connectivity in net/ directory
        - Model equations in eq/ directory
        - Coordinates in coord/ directory (if available)
        - JSON sidecar files with full metadata

        Parameters
        ----------
        output_dir : str
            Root directory for the BIDS dataset.
        subject : str
            Subject identifier (without 'sub-' prefix). Default: '01'.
        session : str, optional
            Session identifier (without 'ses-' prefix).
        description : str
            Description label for the output files. Default: 'tvbsim'.
        run : int, optional
            Run number.
        ts_label : str
            Time series label (e.g., 'sim', 'bold', 'eeg'). Default: 'sim'.
        timeseries : TimeSeries, optional
            Pre-computed TimeSeries. If not provided and run_simulation=True,
            the simulation will be executed.
        run_simulation : bool
            If True and no timeseries provided, run the simulation. Default: True.
        **run_kwargs
            Additional arguments passed to the run() method.

        Returns
        -------
        str
            Path to the created BIDS dataset root directory.

        Examples
        --------
        >>> experiment = SimulationExperiment(...)
        >>> experiment.to_bids("./derivatives/tvbo", subject="01", description="rest")
        './derivatives/tvbo'

        >>> # Or with pre-computed timeseries
        >>> ts = experiment.run()
        >>> experiment.to_bids("./derivatives/tvbo", timeseries=ts)
        './derivatives/tvbo'

        Notes
        -----
        Follows BIDS BEP034 Computational Modeling extension v1.0.0.
        Uses tvbo format for model equations.
        """
        # Run simulation if needed
        if timeseries is None and run_simulation:
            timeseries = self.run(**run_kwargs)

        if timeseries is None:
            raise ValueError(
                "No timeseries provided and run_simulation=False. "
                "Provide a TimeSeries or set run_simulation=True."
            )

        # Delegate to TimeSeries.to_bids with experiment reference
        return timeseries.to_bids(
            output_dir=output_dir,
            subject=subject,
            session=session,
            description=description,
            run=run,
            ts_label=ts_label,
            experiment=self,
        )

    # ---- OpenMINDS JSON-LD conversion ----
    def to_openminds(
        self,
        filepath: str | None = None,
        base_id: str | None = None,
        include_context: bool = True,
    ) -> dict:
        """Export experiment to openMINDS JSON-LD format.

        Parameters
        ----------
        filepath : str, optional
            If provided, write JSON-LD to this file path.
        base_id : str, optional
            Base URI for generating @id values (e.g., "https://example.org/simulations").
        include_context : bool
            Whether to include the @context in the output. Default True.

        Returns
        -------
        dict
            OpenMINDS-compatible JSON-LD dictionary.

        Example
        -------
        >>> exp = SimulationExperiment(...)
        >>> jsonld = exp.to_openminds()
        >>> exp.to_openminds("output.jsonld", base_id="https://example.org")
        """
        from tvbo.export.openminds import experiment_to_openminds, save_openminds

        result = experiment_to_openminds(
            self, base_id=base_id, include_context=include_context
        )

        if filepath:
            save_openminds(self, filepath, base_id=base_id)

        return result

    @classmethod
    def from_openminds(cls, source: str | dict) -> "SimulationExperiment":
        """Create a SimulationExperiment from openMINDS JSON-LD.

        Parameters
        ----------
        source : str or dict
            Either a file path to a JSON-LD file, or a dict containing
            JSON-LD data.

        Returns
        -------
        SimulationExperiment
            New instance constructed from the openMINDS data.

        Example
        -------
        >>> exp = SimulationExperiment.from_openminds("experiment.jsonld")
        >>> exp = SimulationExperiment.from_openminds({"@type": "tvbo:SimulationExperiment", ...})
        """
        from tvbo.export.openminds import experiment_from_openminds, load_openminds

        if isinstance(source, str):
            # It's a file path
            data = load_openminds(source)
        elif isinstance(source, dict):
            data = experiment_from_openminds(source)
        else:
            raise TypeError(f"Expected str or dict, got {type(source)}")

        return cls(**data)
