"""Convert TVB simulator objects to tvbo datamodel instances."""

from typing import Any, Union

from tvbo.datamodel import schema as tvbo_datamodel


def _to_scalar(val):
    """Extract a Python scalar from a numpy array or other array-like."""
    if val is None:
        return None
    if hasattr(val, "item"):
        return val.item() if val.ndim == 0 else val.flat[0]
    if hasattr(val, "__len__") and len(val) > 0:
        return float(val[0])
    return float(val)


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
        p_names = getattr(sim.model, "global_parameter_names", []) + getattr(sim.model, "spatial_parameter_names", [])
        if not p_names:
            raise Exception("No parameters found in model .global_parameter_names and .spatial_parameter_names.")
        for p in p_names:
            try:
                val = getattr(sim.model, p)
                val = val[0] if hasattr(val, "__len__") and len(val) > 0 else val
            except Exception:
                print(f"Failed to parse parameter {p}")
                val = None
            model_metadata.parameters[p] = tvbo_datamodel.Parameter(name=p, value=val)

    except Exception as e:
        print(f"{e} Falling back to summary_info.")
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
            val = getattr(sim.model, k)
            val = val[0] if hasattr(val, "__len__") and len(val) > 0 else val
            model_metadata.parameters[k] = tvbo_datamodel.Parameter(name=k, value=val)

    # State variables with domains/boundaries and optional initial conditions
    for i, sv in enumerate(getattr(sim.model, "state_variables", []) or []):
        lo, hi = getattr(sim.model, "state_variable_range", {}).get(sv, (None, None))
        boundaries = None
        sbb = getattr(sim.model, "state_variable_boundaries", None)
        if sbb and sv in sbb:
            blo, bhi = sbb[sv]
            boundaries = tvbo_datamodel.Range(lo=blo, hi=bhi)

        ics = getattr(sim, "initial_conditions", None)

        model_metadata.state_variables[sv] = tvbo_datamodel.StateVariable(
            name=sv,
            coupling_variable=bool(i in getattr(sim.model, "cvar", []) or []),
            variable_of_interest=bool(sv in getattr(sim.model, "variables_of_interest", []) or []),
            domain=(tvbo_datamodel.Range(lo=float(lo), hi=float(hi)) if (lo is not None or hi is not None) else None),
            boundaries=boundaries,
            initial_value=float(ics[0, i, :, 0].mean()) if ics is not None else None,
        )

    # --- Coupling ---
    coupling_metadata = tvbo_datamodel.Coupling(name=type(sim.coupling).__name__)
    for k, v in dict(getattr(sim.coupling, "summary_info", lambda: {})()).items():
        if k in {"Type", "title", "gid"}:
            continue
        coupling_metadata.parameters[k] = tvbo_datamodel.Parameter(name=k, value=_to_scalar(v))

    # --- Integration ---
    integrator_info = dict(getattr(sim.integrator, "summary_info", lambda: {})())
    integrator_type = integrator_info.get("Type", "")
    method = integrator_type.replace("Stochastic", "").replace("Deterministic", "").strip()

    noise_meta = None
    noise_obj = getattr(sim.integrator, "noise", None)
    if noise_obj is not None:
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
                params["nsig"] = tvbo_datamodel.Parameter(name="nsig", value=nsig_val)

        if nsig.shape[0] > 1:
            for i, sv in enumerate(getattr(sim.model, "state_variables")):
                model_metadata.state_variables[sv].noise = tvbo_datamodel.Noise(
                    parameters={"sigma": {"value": float((2.0 * nsig_arr[i]) ** 0.5)}}
                )

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
        try:
            nseed = getattr(noise_obj, "noise_seed", None)
            if nseed is None:
                rs = getattr(noise_obj, "random_stream", None)
                nseed = getattr(rs, "seed", None) if rs is not None else None
            if nseed is not None:
                params["noise_seed"] = tvbo_datamodel.Parameter(name="noise_seed", value=int(nseed))
        except Exception:
            pass

        additive = False
        try:
            from tvb.simulator.noise import Additive as _Add

            additive = isinstance(noise_obj, _Add)
        except Exception:
            try:
                additive = dict(getattr(noise_obj, "summary_info", lambda: {})()).get("Type") == "Additive"
            except Exception:
                additive = False

        noise_meta = tvbo_datamodel.Noise(parameters=params, additive=additive)

    integration_metadata = tvbo_datamodel.Integrator(method=method, noise=noise_meta)
    if noise_obj is not None:
        try:
            if nsig_arr is None:
                nsig = getattr(noise_obj, "nsig", None)
                if hasattr(nsig, "__len__"):
                    nsig_arr = [float(x) for x in list(nsig)]
            if nsig_arr is not None and len(nsig_arr) > 0:
                s_arr = [float((2.0 * x) ** 0.5) for x in nsig_arr]
                integration_metadata.state_wise_sigma = s_arr
        except Exception:
            pass

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
            label=type(sim.stimulus).__name__,
            weighting=list(getattr(sim.stimulus, "weight", []) or []),
            equation=tvbo_datamodel.Equation(pycode=getattr(getattr(sim.stimulus, "temporal", None), "equation", "")),
            parameters={
                k: tvbo_datamodel.Parameter(name=k, value=_to_scalar(v))
                for k, v in (getattr(getattr(sim.stimulus, "temporal", None), "parameters", {}) or {}).items()
            },
        )
    else:
        stimulus_metadata = None

    # --- Network/Connectome ---
    # Delegate to the canonical from_tvb converter which uses the current
    # Network schema (nodes + set_matrix, not the removed weights/lengths fields)
    from tvbo.data.converters import from_tvb as _from_tvb_connectivity

    network_metadata = _from_tvb_connectivity(sim.connectivity)

    # --- Assemble experiment ---
    exp = tvbo_datamodel.SimulationExperiment(
        id=experiment_id,
        dynamics=model_metadata,
        coupling=coupling_metadata,
        integration=integration_metadata,
        network=network_metadata,
        stimulation=stimulus_metadata,
    )
    # Software environment
    try:
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
        exp.environment = env
    except Exception:
        pass

    # Monitors → Observations (Monitor class no longer exists in schema)
    for mon in getattr(sim, "monitors", []) or []:
        info = dict(getattr(mon, "summary_info", lambda: {})())
        if not info:
            continue
        name = info.get("Type") or type(mon).__name__
        period = _to_scalar(info.get("period"))
        exp.observations[name] = tvbo_datamodel.Observation(name=name, period=period)
    return exp
