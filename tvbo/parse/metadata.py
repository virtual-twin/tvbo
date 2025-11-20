from os.path import isfile, join

import pandas as pd
import yaml
from linkml_runtime.loaders import yaml_loader

from typing import Any, Callable, Dict, List, Sequence, Union
from tvbo.datamodel import tvbo_datamodel
from tvbo.export import experiment
from tvbo.knowledge import ontology, study
from tvbo.knowledge.simulation import equations


def load_simulation_metadata(filename: str) -> Any:
    sim_metadata = tvbo_datamodel.SimulationStudy(**yaml.safe_load(open(filename)))

    return sim_metadata


def load_experiment_metadata(filename: str) -> Any:
    return yaml_loader.load(filename, tvbo_datamodel.SimulationExperiment)


def load_simulation_study(yaml_file: str) -> Any:
    study_metadata = yaml_loader.load(
        yaml_file,
        tvbo_datamodel.SimulationStudy,
    )
    return study.SimulationStudy(study_metadata=study_metadata)


def load_simulation_experiment(yaml_file: str, experiment_id: int = 1) -> Any:
    experiment_metadata = yaml_loader.load(
        yaml_file,
        tvbo_datamodel.SimulationExperiment,
    )

    exp = experiment.SimulationExperiment(**experiment_metadata._as_dict)
    return exp


def import_yaml_model(
    experiment_metadata: Any,
    model_name: Union[str, None] = None,
) -> Any:
    """Import a model from metadata into the ontology.

    Loads model information and creates ontology subclasses for state variables,
    parameters, derived variables, and output transforms.

    Args:
        experiment_metadata: A SimulationExperiment, Dynamics, or path to a YAML file
            from which to read the model metadata.
        model_name (str, optional): Name to use for the created ontology class. If None,
            the name is derived from the metadata. Defaults to None.

    Returns:
        owlready2.entity.ThingClass: The created ontology model class.
    """
    model_data: Any = None
    if isinstance(experiment_metadata, str) and isfile(experiment_metadata):
        experiment_metadata = load_experiment_metadata(experiment_metadata)
        model_data = experiment_metadata.model
    if isinstance(experiment_metadata, tvbo_datamodel.Dynamics):
        model_data = experiment_metadata
    elif isinstance(experiment_metadata, tvbo_datamodel.SimulationExperiment):
        model_data = experiment_metadata.model

    if model_name is None:
        model_name = str(model_data.name)

    onto = ontology.onto

    if ontoclass := onto.search_one(label=model_name):
        return ontoclass

    acr = ontology.create_acronym(model_name)
    model_suffix = f"_{acr}"

    def create_onto_subclass(name, base_class, properties, model_class):
        """
        Helper function to create ontology subclasses.
        """
        with onto:
            new_class = type(name, (model_class, base_class), {})
            for prop_name, prop_value in properties.items():
                if prop_value is not None:
                    getattr(new_class, prop_name).append(prop_value)
            return new_class

    # Create the main model class in the ontology
    with onto:
        model_class = type(
            model_name,
            (onto.NeuralMassModel,),
            {
                "label": model_name,
                "definition": model_name,
                "acronym": acr,
            },
        )

    # Adding state variables using attribute access
    for sv in model_data.state_variables.values():
        properties = {
            "label": sv.name + model_suffix,
            "symbol": str(sv.name),
            "stateVariableRange": (
                f"lo={sv.domain.lo}, hi={sv.domain.hi}" if sv.domain else ""
            ),
        }
        if sv.boundaries:
            properties["stateVariableBoundaries"] = (
                f"lo={sv.boundaries.lo}, hi={sv.boundaries.hi}"
            )

        sv_class = create_onto_subclass(
            sv.name + model_suffix, onto.StateVariable, properties, model_class
        )
        if sv.coupling_variable:
            model_class.has_cvar.append(sv_class)

        # Handle both continuous models (with rhs) and discrete maps (with conditionals)
        if sv.equation.rhs:
            # Continuous model: create TimeDerivative
            td_name = sv.name + "_dot" + model_suffix
            td_class = create_onto_subclass(
                td_name,
                onto.TimeDerivative,
                {
                    "label": td_name,
                    "value": str(sv.equation.rhs),
                    "symbol": str(sv.equation.lhs) if sv.equation.lhs else sv.name,
                },
                sv_class,
            )
        elif sv.equation.conditionals:
            # Discrete map: Don't create a separate ConditionalDerivedVariable
            # The conditionals are part of the state variable's equation and will be
            # handled directly in get_equations() via conditionals2piecewise
            # Just store a marker that this SV has conditionals (optional)
            pass

        with onto:
            model_class.has_state_variable.append(sv_class)
    # Adding parameters using attribute access
    for k, p in model_data.parameters.items():
        properties = {
            "label": k + model_suffix,
            "symbol": getattr(p, "symbol", str(k)),
            "definition": str(p.description),
            "defaultValue": (
                float(p.value) if not isinstance(p.value, list) else p.default
            ),
            "range": (
                f"lo={p.domain.lo}, hi={p.domain.hi}, step={p.domain.step}"
                if p.domain
                else ""
            ),
        }
        p_class = create_onto_subclass(
            k + model_suffix, onto.Parameter, properties, model_class
        )

        model_class.has_parameter.append(p_class)

    # Adding derived variables using attribute access
    for dp in model_data.derived_parameters.values():
        properties = {
            "label": dp.name + model_suffix,
            "equation": str(dp.equation.rhs),
            "value": str(dp.equation.rhs),
            "symbol": str(dp.equation.lhs if dp.equation.lhs else dp.name),
        }
        create_onto_subclass(
            dp.name + model_suffix, onto.Function, properties, model_class
        )

    for dv in model_data.derived_variables.values():
        properties = {
            "label": dv.name + model_suffix,
            "equation": str(dv.equation.rhs),
            "value": str(dv.equation.rhs),
            "symbol": str(dv.equation.lhs if dv.equation.lhs else dv.name),
        }
        create_onto_subclass(
            dv.name + model_suffix, onto.Function, properties, model_class
        )

    for ot in model_data.output_transforms.values():
        properties = {
            "label": ot.name + model_suffix,
            "equation": str(ot.equation.rhs),
            "value": str(ot.equation.rhs),
            "symbol": str(ot.equation.lhs if ot.equation.lhs else ot.name),
        }
        create_onto_subclass(
            ot.name + model_suffix, onto.Function, properties, model_class
        )

    for k, cterm in model_data.coupling_terms.items():
        c_class = create_onto_subclass(
            str(k),
            onto.CouplingTerm,
            {"label": str(k)},
            model_class,
        )
        c_class.is_a.append(onto.GlobalConnectivity)

    equations.update_mathematical_relationships(model_class)

    if references := model_data.has_reference:
        try:
            references = eval(references)
        except:
            if isinstance(references, str):
                references = [references]
        with onto:
            for ref in references:
                if isinstance(ref, str):
                    ref = ontology.onto.search_one(label=str(ref))
                if ref is not None:
                    model_class.has_reference.append(ref)

    return model_class


def import_yaml_coupling(
    experiment_metadata: Any,
    coupling_name: Union[str, None] = None,
) -> Any:
    """Import a coupling function from metadata into the ontology.

    Args:
        experiment_metadata: A SimulationExperiment, Coupling, dict, or path to a YAML file
            from which to read the coupling metadata.
        coupling_name (str, optional): Name to use for the created ontology class. If None,
            the name is derived from the metadata. Defaults to None.

    Returns:
        owlready2.entity.ThingClass: The created ontology coupling class.
    """
    coupling_data: Any = None
    if isinstance(experiment_metadata, str) and isfile(experiment_metadata):
        experiment_metadata = load_experiment_metadata(experiment_metadata)
        coupling_data = experiment_metadata.coupling
    elif isinstance(experiment_metadata, dict):
        experiment_metadata = tvbo_datamodel.SimulationExperiment(**experiment_metadata)
        coupling_data = experiment_metadata.coupling
    elif isinstance(experiment_metadata, tvbo_datamodel.Coupling):
        coupling_data = experiment_metadata
    elif isinstance(experiment_metadata, tvbo_datamodel.SimulationExperiment):
        coupling_data = experiment_metadata.coupling

    if coupling_name is None:
        coupling_name = str(coupling_data.name)

    onto = ontology.onto
    acr = ontology.create_acronym(coupling_name)
    coupling_suffix = f"_{acr}"

    def create_onto_subclass(name, base_class, coupling_class, properties):
        """
        Helper function to create ontology subclasses.
        """
        with onto:
            new_class = type(name, (coupling_class, base_class), {})
            for prop_name, prop_value in properties.items():
                getattr(new_class, prop_name).append(prop_value)
            return new_class

    with onto:
        coupling_class = type(
            coupling_name,
            (onto.Coupling,),
            {
                "label": coupling_name,
                "definition": coupling_name,
                "acronym": acr,
            },
        )

    for k, p in coupling_data.parameters.items():
        p_class = create_onto_subclass(
            p.name,
            onto.Parameter,
            coupling_class,
            {
                "label": p.name + coupling_suffix,
                "description": str(p.description),
                "symbol": str(p.name),
                "defaultValue": float(p.value),
                "range": (
                    f"lo={p.domain.lo}, hi={p.domain.hi}, step={p.domain.step}"
                    if p.domain
                    else ""
                ),
            },
        )
        coupling_class.has_parameter.append(p_class)

    if "pre_expression" in list(coupling_data._keys()) and coupling_data.pre_expression:
        create_onto_subclass(
            "pre" + coupling_suffix,
            onto.Fpre,
            coupling_class,
            {"value": str(coupling_data.pre_expression.rhs)},
        )
    if (
        "post_expression" in list(coupling_data._keys())
        and coupling_data.post_expression
    ):
        create_onto_subclass(
            "post" + coupling_suffix,
            onto.Fpost,
            coupling_class,
            {"value": str(coupling_data.post_expression.rhs)},
        )
    return coupling_class


################################################################################
# TVB to Metadata
################################################################################


def simulator2metadata(sim: Any, experiment_id: Union[int, None] = None, odir: Union[str, None] = None) -> Any:
    """Convert a TVB simulator into a tvbo_datamodel.SimulationExperiment.

    Aligns with the schema-only runtime design: uses datamodel classes directly and
    populates only schema fields.
    """
    # Pick an experiment id if not provided
    if not experiment_id:
        try:
            experiment_id = sim.gid.int
        except Exception:
            experiment_id = 1

    model_metadata = tvbo_datamodel.Dynamics(
        name=type(sim.model).__name__,
        number_of_modes=getattr(sim.model, "number_of_modes", 1),
    )

    # Parameters: prefer explicit TVB lists, fallback to summary_info
    try:
        for p in getattr(sim.model, "global_parameter_names", []) or []:
            try:
                val = getattr(sim.model, p)
                val = val[0] if hasattr(val, "__len__") and len(val) > 0 else val
            except Exception:
                val = None
            model_metadata.parameters[p] = tvbo_datamodel.Parameter(name=p, value=val)

        for p in getattr(sim.model, "local_parameter_names", []) or []:
            model_metadata.parameters[p] = tvbo_datamodel.Parameter(
                name=p, value=getattr(sim.model, p)
            )
    except Exception:
        for k, v in dict(getattr(sim.model, "summary_info", lambda: {})()).items():
            if k in {
                "Type",
                "title",
                "state_variable_range",
                "state_variable_boundaries",
                "variables_of_interest",
                "gid",
            }:
                continue
            model_metadata.parameters[k] = tvbo_datamodel.Parameter(name=k, value=v)

    # State variables with domains/boundaries and optional initial conditions
    for i, sv in enumerate(getattr(sim.model, "state_variables", []) or []):
        lo, hi = getattr(sim.model, "state_variable_range", {}).get(sv, (None, None))
        boundaries = None
        sbb = getattr(sim.model, "state_variable_boundaries", None)
        if sbb and sv in sbb:
            blo, bhi = sbb[sv]
            boundaries = tvbo_datamodel.Range(lo=blo, hi=bhi)

        init_vals = None
        ics = getattr(sim, "initial_conditions", None)
        if ics is not None:
            init_vals = list(float(ic) for ic in ics[0, i, :, 0])

        model_metadata.state_variables[sv] = tvbo_datamodel.StateVariable(
            name=sv,
            coupling_variable=bool(i in getattr(sim.model, "cvar", []) or []),
            variable_of_interest=bool(
                sv in getattr(sim.model, "variables_of_interest", []) or []
            ),
            domain=(
                tvbo_datamodel.Range(lo=float(lo), hi=float(hi))
                if (lo is not None or hi is not None)
                else None
            ),
            boundaries=boundaries,
            initial_value=float(ics[0, i, :, 0].mean()) if ics is not None else None,
            initial_conditions=init_vals,
        )

    # --- Coupling ---
    coupling_metadata = tvbo_datamodel.Coupling(name=type(sim.coupling).__name__)
    for k, v in dict(getattr(sim.coupling, "summary_info", lambda: {})()).items():
        if k in {"Type", "title", "gid"}:
            continue
        coupling_metadata.parameters[k] = tvbo_datamodel.Parameter(name=k, value=v)

    # --- Integration ---
    integrator_info = dict(getattr(sim.integrator, "summary_info", lambda: {})())
    integrator_type = integrator_info.get("Type", "")
    method = (
        integrator_type.replace("Stochastic", "").replace("Deterministic", "").strip()
    )

    noise_meta = None
    noise_obj = getattr(sim.integrator, "noise", None)
    if noise_obj is not None:
        # Prefer nsig; include tau/ntau; include RNG seed where available
        params = {}
        nsig_val = None
        nsig_arr = None
        nsig = getattr(noise_obj, "nsig", None)
        if nsig is not None:
            if hasattr(nsig, "__len__"):
                nsig_arr = [float(x) for x in list(nsig)]
                nsig_val = float(nsig_arr[0]) if nsig_arr else None
            else:
                nsig_val = float(nsig)
            if nsig_val is not None:
                params["nsig"] = tvbo_datamodel.Parameter(
                    name="nsig", value=nsig_val
                )

        if nsig.shape[0] > 1:
            for i, sv in enumerate(getattr(sim.model, "state_variables")):
                model_metadata.state_variables[sv].noise = tvbo_datamodel.Noise(parameters={'sigma':{'value':float((2.0 * nsig_arr[i]) ** 0.5)}})


        for key in ("tau", "ntau"):
            try:
                tv = getattr(noise_obj, key, None)
                if tv is not None:
                    try:
                        tval = float(tv)
                    except Exception:
                        tval = tv
                    params[key] = tvbo_datamodel.Parameter(name=key, value=tval)
            except Exception:
                pass
        # Seed/random stream (map to parameter name expected by templates)
        try:
            nseed = getattr(noise_obj, "noise_seed", None)
            if nseed is None:
                # Sometimes seed might be under random_stream or similar
                rs = getattr(noise_obj, "random_stream", None)
                nseed = getattr(rs, "seed", None) if rs is not None else None
            if nseed is not None:
                params["noise_seed"] = tvbo_datamodel.Parameter(
                    name="noise_seed", value=int(nseed)
                )
        except Exception:
            pass

        # Determine additive vs multiplicative
        additive = False
        try:
            from tvb.simulator.noise import Additive as _Add, Multiplicative as _Mul

            additive = isinstance(noise_obj, _Add)
        except Exception:
            try:
                additive = (
                    dict(getattr(noise_obj, "summary_info", lambda: {})()).get("Type")
                    == "Additive"
                )
            except Exception:
                additive = False

        noise_meta = tvbo_datamodel.Noise(parameters=params, additive=additive)

    integration_metadata = tvbo_datamodel.Integrator(method=method, noise=noise_meta)
    # If nsig array was detected, store state-wise sigma for round-trip fidelity
    if noise_obj is not None:
        try:
            if nsig_arr is None:
                nsig = getattr(noise_obj, "nsig", None)
                if hasattr(nsig, "__len__"):
                    nsig_arr = [float(x) for x in list(nsig)]
            if nsig_arr is not None and len(nsig_arr) > 0:
                # sigma = sqrt(2*nsig)
                s_arr = [float((2.0 * x) ** 0.5) for x in nsig_arr]
                integration_metadata.state_wise_sigma = s_arr
        except Exception:
            pass

    # TVB integrators expose dt; Simulator exposes simulation_length (ms)
    try:
        dt = getattr(sim.integrator, "dt", None)
        if dt is not None:
            integration_metadata.step_size = float(dt)
    except Exception:
        pass
    try:
        simlen = getattr(sim, "simulation_length", None)
        if simlen is not None:
            integration_metadata.duration = float(simlen)
    except Exception:
        pass

    # --- Stimulus ---
    if getattr(sim, "stimulus", None):
        stimulus_metadata = tvbo_datamodel.Stimulus(
            name="Stimulus",
            weighting=list(getattr(sim.stimulus, "weight", []) or []),
            equation=tvbo_datamodel.Equation(
                pycode=getattr(getattr(sim.stimulus, "temporal", None), "equation", "")
            ),
            parameters={
                k: tvbo_datamodel.Parameter(name=k, value=v)
                for k, v in (
                    getattr(getattr(sim.stimulus, "temporal", None), "parameters", {})
                    or {}
                ).items()
            },
        )
    else:
        stimulus_metadata = None

    # --- Network/Connectome ---
    if odir:
        pd.DataFrame(sim.connectivity.weights).to_csv(
            join(odir, f"exp-{experiment_id}_desc-connectome_weights.csv")
        )
        pd.DataFrame(sim.connectivity.tract_lengths).to_csv(
            join(odir, f"exp-{experiment_id}_desc-connectome_lengths.csv")
        )

    network_metadata = tvbo_datamodel.Connectome(
        number_of_regions=sim.connectivity.number_of_regions,
        weights=(
            tvbo_datamodel.Matrix(
                dataLocation=f"exp-{experiment_id}_desc-connectome_weights.csv"
            )
            if odir
            else tvbo_datamodel.Matrix(
                x=tvbo_datamodel.BrainRegionSeries(
                    values=list(sim.connectivity.region_labels)
                ),
                y=tvbo_datamodel.BrainRegionSeries(
                    values=list(sim.connectivity.region_labels)
                ),
                values=[float(w) for w in sim.connectivity.weights.ravel()],
            )
        ),
        lengths=(
            tvbo_datamodel.Matrix(
                dataLocation=f"exp-{experiment_id}_desc-connectome_lengths.csv"
            )
            if odir
            else tvbo_datamodel.Matrix(
                x=tvbo_datamodel.BrainRegionSeries(
                    values=list(sim.connectivity.region_labels)
                ),
                y=tvbo_datamodel.BrainRegionSeries(
                    values=list(sim.connectivity.region_labels)
                ),
                values=[float(l) for l in sim.connectivity.tract_lengths.ravel()],
            )
        ),
        conduction_speed=tvbo_datamodel.Parameter(
            name="conduction_speed",
            label="v",
            value=(
                float(getattr(sim.connectivity, "speed", None))
                if getattr(sim, "connectivity", None) is not None
                and getattr(sim.connectivity, "speed", None) is not None
                else getattr(sim, "conduction_speed", None)
            ),
            unit="mm/ms",
        ),
    )

    # --- Assemble experiment ---
    exp = tvbo_datamodel.SimulationExperiment(
        id=experiment_id,
        local_dynamics=model_metadata,
        coupling=coupling_metadata,
        integration=integration_metadata,
        network=network_metadata,
        stimulation=stimulus_metadata,
    )
    # Software environment and requirements (TVB version)
    try:
        import sys as _sys
        import platform as _platform

        try:
            import tvb as _tvb

            tvb_version = getattr(_tvb, "__version__", None)
        except Exception:
            tvb_version = None
        env = tvbo_datamodel.SoftwareEnvironment(
            software="TVB",
            version=str(tvb_version) if tvb_version is not None else "unknown",
            platform=_platform.platform(),
        )
        if hasattr(exp, "environment"):
            exp.environment = env
        # Add as a requirement entry if supported
        if hasattr(exp, "requirements"):
            req = tvbo_datamodel.SoftwareRequirement(
                name="tvb",
                version=str(tvb_version) if tvb_version is not None else "unknown",
                environment=env,
            )
            # Ensure it's a list-like
            try:
                exp.requirements.append(req)
            except Exception:
                exp.requirements = [req]
    except Exception:
        pass
    # Monitors
    for mon in getattr(sim, "monitors", []) or []:
        info = dict(getattr(mon, "summary_info", lambda: {})())
        if not info:
            continue
        name = info.get("Type") or type(mon).__name__
        period = info.get("period")
        exp.monitors[name] = tvbo_datamodel.Monitor(name=name, period=period)
    return exp


#### Jax Interface ####
from tvbo.utils import Bunch


def add_to_parameters_collection(key: str, value: tvbo_datamodel.Parameter, path: Sequence[Any], parameters: Bunch) -> None:
    """Adds a value to a Bunch object using the provided path, without inserting a redundant sub-level."""
    current_level = parameters  # Start at the top level of the Bunc
    for part in path:  # Traverse the entire path
        if part == "parameters":
            continue
        if part not in current_level:
            current_level[part] = Bunch()  # Create a new Bunch if not present

        if part != key:
            current_level = current_level[part]  # Move deeper into the nested Bunch
    current_level[key] = value.value


def traverse_metadata(
    metadata: Any,
    target_instance: Any = tvbo_datamodel.Parameter,
    path: Union[List[Any], None] = None,
    callback: Union[Callable[..., None], None] = add_to_parameters_collection,
    callback_kwargs: Dict[str, Any] = {},
    keys_to_exclude: Sequence[str] = (),
) -> None:
    """Recursively traverses the attributes of a metadata object, calling a callback on each Parameter."""
    if path is None:
        path = []

    def _is_datamodel_like(obj) -> bool:
        """Return True if obj is a datamodel instance or a subclass living outside tvbo.datamodel.*.

        This catches runtime wrappers that subclass LinkML datamodel classes (e.g., Coupling wrapper).
        """
        try:
            mod = getattr(type(obj), "__module__", "")
            if mod.startswith("tvbo.datamodel."):
                return True
            for base in type(obj).mro():
                if getattr(base, "__module__", "").startswith("tvbo.datamodel."):
                    return True
        except Exception:
            pass
        return False

    # Check if the current object has a __dict__ (not all objects do)
    if hasattr(metadata, "__dict__"):
        if isinstance(metadata, target_instance):
            if callback:
                callback(path[-1], metadata, path, **callback_kwargs)

        for attr_name, attr_value in metadata.__dict__.items():
            if attr_name in keys_to_exclude or attr_value is None:
                continue

            current_path = path + [attr_name]
            # Traverse datamodel objects and subclasses of datamodel classes
            if _is_datamodel_like(attr_value):
                traverse_metadata(
                    attr_value,
                    target_instance,
                    current_path,
                    callback,
                    callback_kwargs,
                    keys_to_exclude,
                )

            # If the attribute is a list, traverse its items
            elif isinstance(attr_value, list):
                for i, item in enumerate(attr_value):
                    traverse_metadata(
                        item,
                        target_instance,
                        current_path + [i],
                        callback,
                        callback_kwargs,
                        keys_to_exclude,
                    )

            # If the attribute is a dict, traverse its values
            elif isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    traverse_metadata(
                        value,
                        target_instance,
                        current_path + [key],
                        callback,
                        callback_kwargs,
                        keys_to_exclude,
                    )
