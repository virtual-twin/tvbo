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

try:
    from lems.base.util import validate_lems
except ImportError:
    validate_lems = None  # PyLEMS is optional (neuroml extra)

from tvbo import templates
from tvbo.data.tvbo_data.connectomes import Network
from tvbo.data.types import SimulationState, TimeSeries, ExperimentResult
from tvbo.datamodel import tvbo_datamodel
from tvbo.export import templater
from tvbo.export.templater import format_code
from tvbo.knowledge import Coupling, Integrator
from tvbo.knowledge.simulation.continuation import Continuation
from tvbo.knowledge.simulation.localdynamics import Dynamics
from tvbo.knowledge.simulation.network import Coupling, _Network
from tvbo.parse import metadata
from tvbo.utils import Bunch

sessionid = 1


def _sync_network_node_count(net):
    """Sync number_of_nodes/number_of_regions from the nodes list.

    When Network is created via LinkML deserialization + __class__ patching,
    Network.__init__ never runs. This ensures node count is consistent.
    """
    if net.nodes:
        n = len(net.nodes)
        net.number_of_nodes = n
        net.number_of_regions = n
    elif (net.number_of_nodes or 0) > 1 and not net.nodes:
        # number_of_nodes set but no nodes list — create default nodes
        net.nodes = [
            tvbo_datamodel.Node(id=i, label=f"node_{i}")
            for i in range(net.number_of_nodes)
        ]
        net.number_of_regions = net.number_of_nodes


def _upgrade_network_couplings(network, coupling_types=None):
    """Upgrade network.coupling entries to runtime Coupling instances and
    apply ``type``-based database/ontology fill.

    Parameters
    ----------
    network : Network
        The network whose coupling entries should be upgraded.
    coupling_types : dict, optional
        Mapping ``{coupling_key: type_ref}`` extracted from the raw YAML
        before LinkML deserialization.  ``type_ref`` is a coupling function
        name or CURIE (e.g. ``"KuramotoCoupling"`` or
        ``"tvbo:KuramotoCoupling"``).
    """
    coupling_types = coupling_types or {}
    coup_dict = getattr(network, 'coupling', None)
    if not coup_dict:
        return

    for key, coup in coup_dict.items():
        # Upgrade __class__ to runtime Coupling (skip if already runtime)
        if not isinstance(coup, Coupling):
            coup.__class__ = Coupling
        # Apply type-based fill if a type reference was specified
        if key in coupling_types:
            coup.populate_from_type(coupling_types[key])


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

        # Extract non-schema fields from network sub-dicts before parent init.
        # These are popped so the datamodel __init__ doesn't reject unknown kwargs.
        _coupling_types = {}
        net_kw = kwargs.get("network")
        if isinstance(net_kw, dict):
            # Extract `type` from network.coupling entries.
            # `type` references a coupling function name/CURIE.
            coup_dict = net_kw.get("coupling")
            if isinstance(coup_dict, dict):
                for key, val in coup_dict.items():
                    if isinstance(val, dict) and "type" in val:
                        _coupling_types[key] = val.pop("type")

        # Allow coupling to be specified as a plain string name (e.g. "KuramotoCoupling")
        # This implies "load from ontology/database", so we flag it for use_ontology
        _coupling_from_name = False
        if "coupling" in kwargs and isinstance(kwargs["coupling"], str):
            kwargs["coupling"] = {"name": kwargs["coupling"]}
            _coupling_from_name = True

        # Delegate to the parent dataclass initializer for normalization
        super().__init__(**kwargs)

        # Normalize and coerce fields while preserving original conditions
        def _coerce(cls, obj, **extra):
            if isinstance(obj, cls):
                return obj
            if hasattr(obj, "_as_dict"):
                return cls(**obj._as_dict, **extra)
            if isinstance(obj, dict):
                return cls(**obj, **extra)
            return obj

        # Prefer `model` (name string) when `dynamics` is missing
        if getattr(self, "model", None) and not getattr(self, "dynamics", None):
            self.dynamics = Dynamics(name=self.model)

        # Coerce dynamics to enhanced Dynamics class
        if getattr(self, "dynamics", None) and not isinstance(self.dynamics, Dynamics):
            if isinstance(self.dynamics, tvbo_datamodel.Dynamics):
                self.dynamics = Dynamics.from_datamodel(self.dynamics)
            else:
                self.dynamics = _coerce(Dynamics, self.dynamics)

        # Mirror dynamics → model
        if getattr(self, "dynamics", None):
            self.model = self.dynamics

        if getattr(self, "coupling", None) and not isinstance(self.coupling, Coupling):
            self.coupling = _coerce(
                Coupling, self.coupling, use_ontology=_coupling_from_name
            )

        if getattr(self, "integration", None) and not isinstance(
            self.integration, Integrator
        ):
            self.integration = _coerce(Integrator, self.integration)

        # Auto-upgrade continuations to runtime Continuation class
        conts = getattr(self, "continuations", None)
        if conts and isinstance(conts, dict):
            for key, val in conts.items():
                if val is not None and not isinstance(val, Continuation):
                    conts[key] = _coerce(Continuation, val)

        if not getattr(self, "network", None):
            self.network = Network()
        else:
            self.network.__class__ = Network
            # Network.__init__ doesn't run when class is patched, so sync here
            _sync_network_node_count(self.network)

        # Upgrade network.coupling entries to runtime Coupling + apply type fills
        _upgrade_network_couplings(self.network, _coupling_types)

        # If network defines couplings and experiment has no explicit coupling,
        # use the first network coupling as the experiment-level default.
        # This ensures the JAX template (which reads experiment.coupling) picks
        # up the right coupling function.
        net_coup = getattr(self.network, 'coupling', None)
        if net_coup and len(net_coup) > 0:
            first_coup = next(iter(net_coup.values()))
            exp_coup = getattr(self, 'coupling', None)
            # Override only if experiment coupling is the auto-generated default
            if exp_coup is None or (
                isinstance(exp_coup, Coupling)
                and str(getattr(exp_coup, 'name', '')) == 'Linear'
                and not _coupling_from_name
            ):
                self.coupling = first_coup

        # Get source file path if loading from file (set by from_file classmethod)
        self._source_file = getattr(self.__class__, '_pending_source_file', None)

        # Load network from BIDS if bids_dir is specified
        if hasattr(self.network, "bids_dir") and self.network.bids_dir:
            self._load_network_from_bids()

        if not getattr(self, "integration", None):
            self.integration = Integrator(method="Heun")

        if not getattr(self, "coupling", None):
            self.coupling = Coupling(name="Linear", use_ontology=True)

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
        """Create from a datamodel instance by copying its already-normalized
        state.

        This avoids the ``_as_dict`` → re-init round-trip which breaks on
        ``inlined_as_dict`` fields (the keyed dict is not valid ``**kwargs``
        for the inner class constructor).  Instead we directly copy the
        ``__dict__`` from the fully-normalised LinkML object and then set
        the convenience aliases that ``__init__`` would normally provide.
        """
        obj = cls.__new__(cls)
        # Copy all already-normalized state from the datamodel instance
        obj.__dict__.update(dm.__dict__)

        # -- Upgrade Dynamics via __class__ reassignment --
        dyn = getattr(obj, "dynamics", None)
        if isinstance(dyn, dict) and dyn:
            for v in dyn.values():
                if isinstance(v, tvbo_datamodel.Dynamics) \
                        and not isinstance(v, Dynamics):
                    v.__class__ = Dynamics
                    v._ontology_class = None
                    v.update_metadata()
                    v.calculate_derived_parameters()
            first = next(iter(dyn.values()))
            obj.__dict__["dynamics"] = first
            obj.__dict__["model"] = first
        elif isinstance(dyn, tvbo_datamodel.Dynamics):
            if not isinstance(dyn, Dynamics):
                dyn.__class__ = Dynamics
                dyn._ontology_class = None
                dyn.update_metadata()
                dyn.calculate_derived_parameters()
            obj.__dict__["model"] = dyn
        else:
            obj.__dict__.setdefault("dynamics", None)

        # -- Upgrade Integrator via __class__ reassignment --
        integ = getattr(obj, "integration", None)
        if integ is not None and not isinstance(integ, Integrator):
            integ.__class__ = Integrator
            integ._populate_from_ontology()
        if not getattr(obj, "integration", None):
            obj.__dict__["integration"] = Integrator(method="Heun")

        # -- Upgrade Coupling via __class__ reassignment --
        coup = getattr(obj, "coupling", None)
        if coup is not None and not isinstance(coup, Coupling):
            coup.__class__ = Coupling
            # Only fill from ontology if coupling has name but no expressions
            # (i.e. it's a name-only reference, not a fully-specified coupling)
            if getattr(coup, "name", None) and not getattr(coup, "pre_expression", None):
                coup._populate_from_ontology()
        if not getattr(obj, "coupling", None):
            obj.__dict__["coupling"] = Coupling(name="Linear", use_ontology=True)

        # -- Upgrade Network via __class__ reassignment --
        net = getattr(obj, "network", None)
        if net is not None and not isinstance(net, Network):
            net.__class__ = Network
            _sync_network_node_count(net)
            if not getattr(net, "conduction_speed", None):
                net.conduction_speed = tvbo_datamodel.Parameter(
                    name="conduction_speed", label="v",
                    value=3.0, unit="mm/ms",
                )
        if not getattr(obj, "network", None):
            obj.__dict__["network"] = Network()

        # Upgrade network coupling entries (no type refs in from_datamodel path)
        _upgrade_network_couplings(obj.network)

        obj.__dict__["_source_file"] = None
        return obj

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
            New instance with primary dynamics and network.dynamics for
            multi-operator files.

        Example
        -------
        >>> exp = SimulationExperiment.from_pyrates("synaptic_plasticity.yaml")
        >>> print(exp.dynamics.name)  # 'tsodyks'
        >>> print(list(exp.network.dynamics.keys()))  # ['tsodyks', 'depression', 'facilitation']
        """
        from tvbo.export.pyrates import from_pyrates_yaml_all

        dynamics_dicts = from_pyrates_yaml_all(filepath)

        # Convert dicts to Dynamics instances using lightweight construction
        dynamics = {}
        for name, dyn_dict in dynamics_dicts.items():
            # Create instance with _skip_ontology=True to avoid slow lookups
            dyn = Dynamics(_skip_ontology=True, **dyn_dict)
            dynamics[name] = dyn

        # Use the first dynamics as the primary model
        primary_dyn = None
        if dynamics:
            first_key = next(iter(dynamics))
            primary_dyn = dynamics[first_key]

        # Store all dynamics in network.dynamics for heterogeneous support
        network_kwargs = {}
        if len(dynamics) > 1:
            network_kwargs["dynamics"] = dynamics

        return cls(
            dynamics=primary_dyn,
            network=Network(**network_kwargs) if network_kwargs else None,
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
        ... dynamics:
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
        >>> print(exp.dynamics.name)
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
        dynamics = None
        if bids_data["equations"] is not None:
            eq_data = bids_data["equations"]
            model_type = eq_data.get("model_type", "Generic2dOscillator")
            params = eq_data.get("parameters", {})

            # Try to load from ontology by name
            try:
                dynamics = Dynamics.from_ontology(model_type)
                # Apply stored parameters
                for param_name, param_value in params.items():
                    if (
                        hasattr(dynamics, "parameters")
                        and param_name in dynamics.parameters
                    ):
                        dynamics.parameters[param_name].value = param_value
            except Exception:
                # Fallback: create minimal Dynamics
                dynamics = Dynamics(name=model_type)
        else:
            # Default dynamics
            dynamics = Dynamics.from_ontology("Generic2dOscillator")

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
            dynamics=dynamics,
            network=network,
            integration=integration,
            coupling=Coupling(name="Linear", use_ontology=True),  # Default coupling
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

    def symbolic(self, integrate=False, indexed=False, delays=False):
        """Symbolic representation of the full experiment equations.

        Produces different styles of mathematical output depending on the
        combination of flags:

        +----------+--------+-------+--------------------------------------------+
        | integrate| indexed| delays| Description                                |
        +==========+========+=======+============================================+
        | False    | False  | False | Dynamics separated, coupling terms as free |
        |          |        |       | symbols.  Coupling equations shown in      |
        |          |        |       | ``'coupling'`` dict.                       |
        +----------+--------+-------+--------------------------------------------+
        | True     | False  | False | Coupling substituted into dynamics.        |
        |          |        |       | State vars remain ``y0(t)``.               |
        +----------+--------+-------+--------------------------------------------+
        | False    | True   | False | State vars indexed ``y0_i(t)``.            |
        |          |        |       | Coupling shown separately with ``[i],[j]``. |
        +----------+--------+-------+--------------------------------------------+
        | True     | True   | False | Fully integrated with node indices.        |
        |          |        |       | Ready for network presentation.            |
        +----------+--------+-------+--------------------------------------------+
        | *        | (True) | True  | Like above but incoming states carry       |
        |          |        |       | ``y1[j, t - tau[i,j]]`` time delay.        |
        |          |        |       | ``delays=True`` implies ``indexed=True``.  |
        +----------+--------+-------+--------------------------------------------+

        Parameters
        ----------
        integrate : bool
            Substitute coupling expressions into state equations.
        indexed : bool
            Add node index ``_i`` to state / derived variables.
        delays : bool
            Show time delays on incoming coupling states.
            Implies ``indexed=True``.

        Returns
        -------
        dict
            Keys: ``'state'``, ``'coupling'``, ``'functions'``,
            ``'derived_parameters'``, ``'derived'``, ``'parameters'``.
        """
        import sympy as sp
        from sympy import Symbol, Function

        if delays:
            indexed = True

        # Start from local dynamics
        dyn_sym = self.dynamics.symbolic

        # Collect coupling term → Coupling object mapping
        coupling_map = {}
        net_coup = getattr(self.network, "coupling", {}) or {}
        dyn_ct = getattr(self.dynamics, "coupling_terms", {}) or {}

        for ct_name in dyn_ct:
            if ct_name in net_coup:
                coupling_map[ct_name] = net_coup[ct_name]
            elif self.coupling and str(getattr(self.coupling, "name", "")) != "Linear":
                # Fallback: single experiment-level coupling for any term
                coupling_map[ct_name] = self.coupling

        # If no coupling to resolve, return dynamics as-is
        if not coupling_map:
            dyn_sym["coupling"] = {}
            return dyn_sym

        # Build coupling symbolic expressions
        t = Symbol("t")
        coupling_exprs = {}
        for ct_name, coup in coupling_map.items():
            coupling_exprs[ct_name] = coup.symbolic(delays=delays)

        # Substitution maps (built conditionally)
        subs_index = {}
        subs_coupling = {}

        # Node-index substitution: y0(t) → y0_i(t)
        if indexed:
            for sv_name in self.dynamics.state_variables:
                old_f = Function(str(sv_name))
                new_f = Function(str(sv_name) + "_i")
                subs_index[old_f(t)] = new_f(t)
            for dv_name in getattr(self.dynamics, "derived_variables", {}) or {}:
                old_f = Function(str(dv_name))
                new_f = Function(str(dv_name) + "_i")
                subs_index[old_f(t)] = new_f(t)

        # Coupling substitution: Symbol(ct_name) → Sum(...)
        if integrate:
            for ct_name, expr in coupling_exprs.items():
                subs_coupling[Symbol(str(ct_name))] = expr

        # Apply substitutions to all equation lists
        with sp.evaluate(False):
            def _apply_subs(eq):
                """Apply node indexing + coupling substitution to an Eq."""
                lhs = eq.lhs.subs(subs_index)
                rhs = eq.rhs.subs(subs_index).subs(subs_coupling)
                return sp.Eq(lhs, rhs)

            state_eqs = [_apply_subs(eq) for eq in dyn_sym["state"]]
            dv_eqs = [_apply_subs(eq) for eq in dyn_sym["derived"]]
            dp_eqs = list(dyn_sym["derived_parameters"])  # no state vars here
            func_eqs = list(dyn_sym["functions"])  # no state vars here

        return {
            "state": state_eqs,
            "coupling": coupling_exprs,
            "functions": func_eqs,
            "derived_parameters": dp_eqs,
            "derived": dv_eqs,
            "parameters": dyn_sym["parameters"],
        }

    def noise_sigma_array(self) -> np.ndarray:
        """Per-state-variable noise sigma values.

        Preference order:
        1) sigma from each state variable's noise.parameters["sigma"].value
        2) fallback to integration-level noise.parameters["sigma"].value
        3) default 0.0

        Returns an array with one entry per state variable in model order.
        """
        sigmas: list[float] = []

        for sv in self.dynamics.state_variables.values():
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

            # Build dynamics dict: primary model + any network dynamics
            dynamics = self.dynamics

            # Convert datamodel Dynamics to full Dynamics class with methods
            if dynamics is not None and not hasattr(dynamics, "render_equation"):
                dynamics = DynamicsClass.from_datamodel(dynamics)

            # For network case, build full dynamics dict
            if network is not None:
                dynamics_dict = {}
                if dynamics:
                    dynamics_dict[dynamics.name] = dynamics

                # Add any additional dynamics from network.dynamics
                net_dynamics = getattr(network, 'dynamics', None)
                if isinstance(net_dynamics, dict):
                    for name, dyn in net_dynamics.items():
                        if name not in dynamics_dict:
                            if dyn is not None and not hasattr(dyn, "render_equation"):
                                dynamics_dict[name] = DynamicsClass.from_datamodel(dyn)
                            else:
                                dynamics_dict[name] = dyn

                return to_pyrates_yaml_string(
                    dynamics=dynamics_dict,
                    network=network,
                    filepath=filepath,
                )
            else:
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
            state._svar_names = list(self.dynamics.state_variables.keys())
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
            # Use a fresh namespace each time to avoid JAX tracer leaks
            # between repeated executions (stale tracers in shared globals
            # cause UnexpectedTracerError on re-runs).
            namespace = {"TimeSeries": TimeSeries}
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

    def run(self, format="tvboptim", initial_conditions=None, **kwargs):
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

                # Add timings to results and wrap in ExperimentResult
                results.timings = timings
                return ExperimentResult(results, experiment_name=self.label)
            else:
                raw_results = ns.run_experiment(
                    weights=self.network.weights,
                    distances=self.network.distances,
                    mode=mode,
                    **kwargs,
                )
                return ExperimentResult(raw_results, experiment_name=self.label)

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
            #     "State Variable": list(self.dynamics.state_variables.keys()),
            #     "Region": self.network.labels,
            # }
            # ts.sample_period = self.integration.step_size
            # ts.dt = self.integration.step_size

            # Link TimeSeries to source experiment for provenance tracking
            ts.source_experiment = self

            return ts

        elif format.lower() == "cuda":
            from tvbo.export.cuda import run_cuda
            return run_cuda(self, **kwargs)

        elif format.lower() == "python":
            bnm = _Network(Network(self.network))
            bnm.add_local_model(self.dynamics)
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

        elif format.lower() in ["networkdynamics", "nd", "networkdynamics.jl"]:
            return self._run_networkdynamics(**kwargs)

        elif format.lower() in ["mtk", "modelingtoolkit", "modelingtoolkit.jl"]:
            return self._run_modelingtoolkit(**kwargs)

        elif format.lower() in [
            "bifurcationkit", "bifurcationkit.jl", "bifurcation",
            "bifurcation-julia",
        ]:
            return self._run_bifurcation(**kwargs)

        elif format.lower() in [
            "pyrates-bifurcation", "pyrates-bif", "pycobi",
            "bifurcation-pyrates", "auto", "auto-07p",
        ]:
            return self._run_pyrates_bifurcation(**kwargs)

        elif format.lower() in [
            "julia", "diffeq", "differentialequations",
            "differentialequations.jl",
        ]:
            return self._run_julia(**kwargs)

        else:
            raise ValueError(
                f"Format {format} not supported. Valid formats: tvb, jax, python, pyrates, "
                "networkdynamics, mtk, modelingtoolkit, bifurcationkit.jl, "
                "pyrates-bifurcation, julia"
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
        from tvbo.adapters.pyrates import PyRatesAdapter

        adapter = PyRatesAdapter(self)
        return adapter.run(
            solver=solver,
            inputs=inputs,
            outputs=outputs,
            matrix_edge_threshold=matrix_edge_threshold,
            **kwargs,
        )

    def _run_networkdynamics(self, **kwargs) -> TimeSeries:
        """Run simulation using NetworkDynamics.jl via pyjulia."""
        from tvbo.adapters.networkdynamics import NetworkDynamicsAdapter

        adapter = NetworkDynamicsAdapter(self)
        return adapter.run(**kwargs)

    def _run_modelingtoolkit(self, **kwargs) -> TimeSeries:
        """Run simulation using pure ModelingToolkit.jl via pyjulia."""
        from tvbo.adapters.modelingtoolkit import ModelingToolkitAdapter

        adapter = ModelingToolkitAdapter(self)
        return adapter.run(**kwargs)

    def _run_bifurcation(self, **kwargs):
        """Run bifurcation analysis via BifurcationKit.jl."""
        from tvbo.adapters.bifurcationkit import BifurcationKitAdapter

        adapter = BifurcationKitAdapter(self)
        return adapter.run(**kwargs)

    def _run_pyrates_bifurcation(self, **kwargs):
        """Run bifurcation analysis via PyRates/PyCoBi (AUTO-07p)."""
        from tvbo.adapters.pyrates_bifurcation import PyRatesBifurcationAdapter

        adapter = PyRatesBifurcationAdapter(self)
        return adapter.run(**kwargs)

    def _run_julia(self, **kwargs) -> TimeSeries:
        """Run simulation using DifferentialEquations.jl via juliacall."""
        from tvbo.adapters.diffeq import DiffEqAdapter

        adapter = DiffEqAdapter(self)
        return adapter.run(**kwargs)

    def get_experiment_file_prefix(self):
        atlas = (
            f"_atlas-{self.network.parcellation.atlas.name}"
            if self.network and self.network.parcellation
            else ""
        )
        return f"ses-{self.id}_desc-{self.dynamics.label}"

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
        n_modes = getattr(self.dynamics, 'number_of_modes', 1) or 1
        n_nodes = self.network.number_of_regions

        if random:
            history.append(
                self.dynamics.get_initial_values(random=True, N=n_nodes)
            )
        else:
            for sv in self.dynamics.state_variables.values():
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
        if validate_lems is not None:
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

        model = self.dynamics.to_lems(initial_conditions=initial_conditions)

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

        sv_names = list(self.dynamics.state_variables.keys())
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

        elif format.lower() in ["rateml", "rateml-python"]:
            # RateML-style TVB Python model with Numba gufunc
            template = templates.lookup.get_template(
                "rateml/tvbo-rateml-python.py.mako"
            )
            rendered_code = format_code(
                template.render(model=self.dynamics, experiment=self, **kwargs),
                use_black=False,
            )

        elif format.lower() in ["rateml-cuda", "cuda"]:
            # RateML-style CUDA kernel
            template = templates.lookup.get_template(
                "rateml/tvbo-rateml-cuda.c.mako"
            )
            rendered_code = template.render(
                model=self.dynamics,
                experiment=self,
                coupling=self.coupling,
                **kwargs
            )

        elif format.lower() == "rateml-driver":
            # PyCUDA driver for RateML CUDA kernel
            template = templates.lookup.get_template(
                "rateml/tvbo-rateml-driver.py.mako"
            )
            rendered_code = format_code(
                template.render(model=self.dynamics, experiment=self, **kwargs),
                use_black=False,
            )

        elif format.lower() == "julia":
            template = templates.lookup.get_template(
                "tvbo-julia-DifferentialEquations.jl.mako"
            )
            rendered_code = template.render(
                experiment=self, model=self.dynamics, **kwargs
            )

        elif format.lower() in ["networkdynamics", "nd", "networkdynamics.jl"]:
            from tvbo.adapters.base import BaseAdapter
            adapter = BaseAdapter(self)
            ctx = adapter.prepare_context()
            ctx.update(kwargs)
            template = templates.lookup.get_template(
                "tvbo-nd-experiment.jl.mako"
            )
            rendered_code = template.render(**ctx)

        elif format.lower() in ["mtk", "modelingtoolkit", "modelingtoolkit.jl"]:
            from tvbo.adapters.modelingtoolkit import ModelingToolkitAdapter
            adapter = ModelingToolkitAdapter(self)
            rendered_code = adapter.render_code(**kwargs)

        elif format.lower() in [
            "bifurcationkit", "bifurcationkit.jl", "bifurcation",
            "bifurcation-julia",
        ]:
            from tvbo.adapters.bifurcationkit import BifurcationKitAdapter
            adapter = BifurcationKitAdapter(self)
            rendered_code = adapter.render_code(**kwargs)

        elif format.lower() in [
            "pyrates-bifurcation", "pyrates-bif", "pycobi",
            "bifurcation-pyrates", "auto", "auto-07p",
        ]:
            from tvbo.adapters.pyrates_bifurcation import PyRatesBifurcationAdapter
            adapter = PyRatesBifurcationAdapter(self)
            rendered_code = adapter.render_code(**kwargs)

        else:
            raise ValueError(
                f"Unknown format: {format}. Supported: tvb, autodiff, jax, pde, tvboptim, "
                "rateml, rateml-python, rateml-cuda, cuda, rateml-driver, "
                "julia, networkdynamics, nd, mtk, modelingtoolkit, "
                "bifurcationkit.jl, pyrates-bifurcation"
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
    def report(
        self,
        format: str | None = "markdown",
        template_name: str = "tvbo-report-experiment",
        outputfile: str | None = None,
        derivative_notation: str = "dot",
    ) -> str:
        """Render a human-readable report for this experiment.

        - Reuses the model/dynamics report template via Mako include to avoid redundancy.
        - Summarizes integration, network/connectome, coupling, monitors, stimulation, and software info.

        Parameters
        - format: optional explicit fallback format ('markdown' or 'pdf')
        - template_name: base name of the template without extension
        - outputfile: optional path to write the rendered report;
            when provided, extension defines output format (.md or .pdf)
        """
        normalized_format = format.lower() if isinstance(format, str) else None

        if outputfile:
            ext = os.path.splitext(outputfile)[1].lower()
            ext_to_format = {
                ".md": "markdown",
                ".markdown": "markdown",
                ".pdf": "pdf",
            }
            if ext not in ext_to_format:
                raise ValueError(
                    "outputfile extension must be one of: .md, .pdf"
                )
            normalized_format = ext_to_format[ext]

        if normalized_format is None:
            normalized_format = "markdown"

        if normalized_format not in ["markdown", "md", "pdf"]:
            raise ValueError("format must be one of: markdown, pdf")

        md_template = templates.lookup.get_template(f"report/{template_name}.md.mako")
        md_render = md_template.render(
            experiment=self, derivative_notation=derivative_notation
        )

        render = md_render

        # Persist if requested
        if outputfile:
            if normalized_format in ["pdf"]:
                from tvbo.export import report as _report

                _report.to_pdf(md_render, outputfile)
            else:
                with open(outputfile, "w", encoding="utf-8") as f:
                    f.write(render)

        return render

    def generate_report(
        self,
        format: str | None = "markdown",
        template_name: str = "tvbo-report-experiment",
        outputfile: str | None = None,
        derivative_notation: str = "dot",
    ) -> str:
        """Backward-compatible alias for :meth:`report`."""
        return self.report(
            format=format,
            template_name=template_name,
            outputfile=outputfile,
            derivative_notation=derivative_notation,
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
            suffix=ts_label,
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
