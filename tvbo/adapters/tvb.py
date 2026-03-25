# -*- coding: utf-8 -*-
"""TVB (The Virtual Brain) adapter for tvbo.

All TVB ↔ tvbo conversion logic lives here:

**Import (TVB → tvbo)**

- :func:`from_tvb_simulator` — full Simulator → SimulationExperiment
- :func:`from_tvb` — Connectivity → Network
- :func:`from_tvb_zip` — connectivity ZIP file → Network
- :func:`from_tvb_surface` — Connectivity + Surface + RegionMapping → Network pair

**Export (tvbo → TVB)**

- :func:`to_tvb` — Network → TVB Connectivity
- :func:`to_tvb_monitor` — Observation → TVB Monitor
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Union

from tvbo.datamodel import schema as tvbo_datamodel


def _to_scalar(val):
    """Extract a Python scalar from a numpy array or other array-like."""
    if val is None:
        return None
    if hasattr(val, 'item'):
        return val.item() if val.ndim == 0 else val.flat[0]
    if hasattr(val, '__len__') and len(val) > 0:
        return float(val[0])
    return float(val)


def _extract_dynamics(sim) -> tvbo_datamodel.Dynamics:
    """Extract Dynamics metadata from a TVB model."""
    model_metadata = tvbo_datamodel.Dynamics(
        name=type(sim.model).__name__,
        number_of_modes=getattr(sim.model, "number_of_modes", 1),
    )

    # Parameters: prefer explicit TVB lists, fallback to summary_info
    p_names = (
        list(getattr(sim.model, "global_parameter_names", []) or [])
        + list(getattr(sim.model, "spatial_parameter_names", []) or [])
    )
    if p_names:
        for p in p_names:
            val = getattr(sim.model, p, None)
            if val is not None:
                val = val[0] if hasattr(val, "__len__") and len(val) > 0 else val
            model_metadata.parameters[p] = tvbo_datamodel.Parameter(
                name=p, value=val
            )
    else:
        # Fallback to summary_info for models without parameter name lists
        for k, v in dict(
            getattr(sim.model, "summary_info", lambda: {})()
        ).items():
            if k in {
                "Type", "title", "state_variable_range",
                "state_variable_boundaries", "variables_of_interest", "gid",
            }:
                continue
            val = getattr(sim.model, k)
            val = val[0] if hasattr(val, "__len__") and len(val) > 0 else val
            model_metadata.parameters[k] = tvbo_datamodel.Parameter(
                name=k, value=val
            )

    # State variables
    for i, sv in enumerate(getattr(sim.model, "state_variables", []) or []):
        lo, hi = getattr(sim.model, "state_variable_range", {}).get(
            sv, (None, None)
        )
        boundaries = None
        sbb = getattr(sim.model, "state_variable_boundaries", None)
        if sbb and sv in sbb:
            blo, bhi = sbb[sv]
            boundaries = tvbo_datamodel.Range(lo=blo, hi=bhi)

        ics = getattr(sim, "initial_conditions", None)

        cvar = getattr(sim.model, "cvar", None)
        if cvar is None:
            cvar = []

        voi = getattr(sim.model, "variables_of_interest", None)
        if voi is None:
            voi = []

        model_metadata.state_variables[sv] = tvbo_datamodel.StateVariable(
            name=sv,
            coupling_variable=bool(i in cvar),
            variable_of_interest=bool(sv in voi),
            domain=(
                tvbo_datamodel.Range(lo=float(lo), hi=float(hi))
                if (lo is not None or hi is not None)
                else None
            ),
            boundaries=boundaries,
            initial_value=(
                float(ics[0, i, :, 0].mean()) if ics is not None else None
            ),
        )

    return model_metadata


def _extract_coupling(sim) -> tvbo_datamodel.Coupling:
    """Extract Coupling metadata from a TVB simulator."""
    name = type(sim.coupling).__name__
    coupling = tvbo_datamodel.Coupling(name=name, iri=f"tvbo:{name}")
    for k, v in dict(
        getattr(sim.coupling, "summary_info", lambda: {})()
    ).items():
        if k in {"Type", "title", "gid"}:
            continue
        coupling.parameters[k] = tvbo_datamodel.Parameter(
            name=k, value=_to_scalar(v)
        )
    return coupling


def _extract_noise(sim, model_metadata) -> tvbo_datamodel.Noise | None:
    """Extract Noise metadata from a TVB integrator.

    Also sets per-state-variable noise on model_metadata if nsig is
    state-variable-specific.
    """
    noise_obj = getattr(sim.integrator, "noise", None)
    if noise_obj is None:
        return None

    params = {}
    nsig = getattr(noise_obj, "nsig", None)
    nsig_arr = None
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

        # Per-state-variable noise
        if nsig_arr and len(nsig_arr) > 1:
            for i, sv_name in enumerate(
                getattr(sim.model, "state_variables", [])
            ):
                if sv_name in model_metadata.state_variables:
                    sigma = float((2.0 * nsig_arr[i]) ** 0.5)
                    model_metadata.state_variables[sv_name].noise = (
                        tvbo_datamodel.Noise(
                            parameters={
                                "sigma": tvbo_datamodel.Parameter(
                                    name="sigma", value=sigma,
                                )
                            }
                        )
                    )

    # ntau / tau
    for key in ("tau", "ntau"):
        tv = getattr(noise_obj, key, None)
        if tv is not None:
            params[key] = tvbo_datamodel.Parameter(
                name=key, value=_to_scalar(tv)
            )

    # noise seed
    nseed = getattr(noise_obj, "noise_seed", None)
    if nseed is None:
        rs = getattr(noise_obj, "random_stream", None)
        nseed = getattr(rs, "seed", None) if rs is not None else None
    if nseed is not None:
        params["noise_seed"] = tvbo_datamodel.Parameter(
            name="noise_seed", value=int(nseed)
        )

    # Additive vs multiplicative
    additive = False
    try:
        from tvb.simulator.noise import Additive as _Add
        additive = isinstance(noise_obj, _Add)
    except ImportError:
        info = dict(getattr(noise_obj, "summary_info", lambda: {})())
        additive = info.get("Type") == "Additive"

    return tvbo_datamodel.Noise(parameters=params, additive=additive)


def _extract_integration(sim, model_metadata) -> tvbo_datamodel.Integrator:
    """Extract Integrator metadata from a TVB simulator."""
    integrator_info = dict(
        getattr(sim.integrator, "summary_info", lambda: {})()
    )
    integrator_type = integrator_info.get("Type", "")
    method = (
        integrator_type.replace("Stochastic", "")
        .replace("Deterministic", "")
        .strip()
    )

    noise_meta = _extract_noise(sim, model_metadata)

    integration = tvbo_datamodel.Integrator(method=method, noise=noise_meta)

    # State-wise sigma
    noise_obj = getattr(sim.integrator, "noise", None)
    if noise_obj is not None:
        nsig = getattr(noise_obj, "nsig", None)
        if nsig is not None and hasattr(nsig, "__len__"):
            nsig_arr = [float(x) for x in list(nsig)]
            if nsig_arr:
                integration.state_wise_sigma = [
                    float((2.0 * x) ** 0.5) for x in nsig_arr
                ]

    dt = getattr(sim.integrator, "dt", None)
    if dt is not None:
        integration.step_size = float(dt)

    simlen = getattr(sim, "simulation_length", None)
    if simlen is not None:
        integration.duration = float(simlen)

    return integration


def _extract_stimulus(sim) -> tvbo_datamodel.Stimulus | None:
    """Extract Stimulus metadata from a TVB simulator."""
    if not getattr(sim, "stimulus", None):
        return None

    temporal = getattr(sim.stimulus, "temporal", None)
    return tvbo_datamodel.Stimulus(
        label=type(sim.stimulus).__name__,
        weighting=list(getattr(sim.stimulus, "weight", []) or []),
        equation=tvbo_datamodel.Equation(
            pycode=getattr(temporal, "equation", "") if temporal else "",
        ),
        parameters={
            k: tvbo_datamodel.Parameter(name=k, value=_to_scalar(v))
            for k, v in (
                getattr(temporal, "parameters", {}) or {}
            ).items()
        },
    )


def _extract_observations(sim) -> dict:
    """Extract monitors as Observations from a TVB simulator.

    Maps each TVB monitor to a tvbo Observation with full metadata
    including ``class_reference`` for exact TVB class reconstruction.
    """
    from tvb.simulator import monitors as tvb_monitors
    from tvb.datatypes import equations as tvb_equations

    # Map TVB monitor class → (tvbo observation name, TVB class name)
    _MONITOR_MAP = {
        tvb_monitors.Raw: ("Raw", "Raw"),
        tvb_monitors.RawVoi: ("Raw", "Raw"),
        tvb_monitors.SubSample: ("SubSample", "SubSample"),
        tvb_monitors.TemporalAverage: ("TemporalAverage", "TemporalAverage"),
        tvb_monitors.GlobalAverage: ("GlobalAverage", "GlobalAverage"),
        tvb_monitors.SpatialAverage: ("SpatialAverage", "SpatialAverage"),
        tvb_monitors.EEG: ("EEG", "EEG"),
        tvb_monitors.MEG: ("MEG", "MEG"),
        tvb_monitors.iEEG: ("iEEG", "iEEG"),
        tvb_monitors.Bold: ("BOLD_TVB", "Bold"),
        tvb_monitors.BoldRegionROI: ("BOLD_RegionROI", "BoldRegionROI"),
        tvb_monitors.AfferentCoupling: ("AfferentCoupling", "AfferentCoupling"),
        tvb_monitors.AfferentCouplingTemporalAverage: (
            "AfferentCouplingTemporalAverage",
            "AfferentCouplingTemporalAverage",
        ),
    }

    # Map HRF kernel class → (tvbo observation name, HRF class name)
    _HRF_MAP = {
        tvb_equations.FirstOrderVolterra: ("BOLD_TVB", "FirstOrderVolterra"),
        tvb_equations.Gamma: ("BOLD_Gamma", "Gamma"),
        tvb_equations.DoubleExponential: ("BOLD_DoubleExponential", "DoubleExponential"),
        tvb_equations.MixtureOfGammas: ("BOLD_MixtureOfGammas", "MixtureOfGammas"),
    }

    observations = {}
    for mon in getattr(sim, "monitors", []) or []:
        mon_class = type(mon)
        entry = _MONITOR_MAP.get(mon_class)
        if entry is None:
            obs_name = mon_class.__name__
            tvb_class_name = mon_class.__name__
        else:
            obs_name, tvb_class_name = entry

        # For BOLD monitors, select based on HRF kernel type
        hrf_class_name = None
        if isinstance(mon, tvb_monitors.Bold) and hasattr(mon, "hrf_kernel"):
            hrf_entry = _HRF_MAP.get(type(mon.hrf_kernel))
            if hrf_entry:
                obs_name, hrf_class_name = hrf_entry

        period = _to_scalar(getattr(mon, "period", None))
        voi = getattr(mon, "variables_of_interest", None)

        obs_kwargs = dict(name=obs_name, period=period)

        # Imaging modality
        if isinstance(mon, tvb_monitors.Bold):
            obs_kwargs["imaging_modality"] = "BOLD"
        elif isinstance(mon, tvb_monitors.EEG):
            obs_kwargs["imaging_modality"] = "EEG"
        elif isinstance(mon, tvb_monitors.MEG):
            obs_kwargs["imaging_modality"] = "MEG"
        elif isinstance(mon, tvb_monitors.iEEG):
            obs_kwargs["imaging_modality"] = "SEEG"

        # VOI
        if voi is not None and hasattr(voi, '__len__') and len(voi) > 0:
            obs_kwargs["voi"] = int(voi[0])

        # --- Build class_reference for exact TVB class reconstruction ---
        cr_args = []

        # BOLD HRF kernel as constructor arg
        if isinstance(mon, tvb_monitors.Bold):
            if hrf_class_name:
                cr_args.append(tvbo_datamodel.Argument(
                    name="hrf_kernel", value=hrf_class_name,
                    description=f"tvb.datatypes.equations.{hrf_class_name}",
                ))
            hrf_length = _to_scalar(getattr(mon, "hrf_length", 20000.0))
            cr_args.append(tvbo_datamodel.Argument(
                name="hrf_length", value=hrf_length,
            ))

            # Store HRF parameters in pipeline
            hrf = mon.hrf_kernel
            hrf_params = hrf.parameters if hasattr(hrf, "parameters") else {}
            hrf_length_s = hrf_length / 1000.0 if hrf_length else 20.0

            hrf_step = tvbo_datamodel.FunctionCall(
                name="hrf_kernel",
                acronym="HRF",
                output="hrf_kernel",
                equation=tvbo_datamodel.Equation(rhs=str(hrf.equation)),
                arguments=[
                    tvbo_datamodel.Argument(name=k, value=float(v))
                    for k, v in hrf_params.items()
                ],
                time_range=tvbo_datamodel.Range(
                    lo=0.0, hi=hrf_length_s, step=0.004
                ),
            )
            conv_step = tvbo_datamodel.FunctionCall(
                name="fftconvolve",
                input="temporal_average_interim",
                output="convolved",
                callable=tvbo_datamodel.Callable(
                    name="fftconvolve", module="scipy.signal"
                ),
            )
            pipeline = [hrf_step, conv_step]
            if isinstance(hrf, tvb_equations.FirstOrderVolterra):
                k_1 = float(hrf_params.get("k_1", 5.6))
                V_0 = float(hrf_params.get("V_0", 0.02))
                pipeline.append(tvbo_datamodel.FunctionCall(
                    name="volterra_transform",
                    input="convolved",
                    output="Bold",
                    equation=tvbo_datamodel.Equation(
                        rhs=f"(X - 1.0) * {k_1} * {V_0}"
                    ),
                ))
            obs_kwargs["pipeline"] = pipeline

        # Projection monitor parameters (EEG, MEG, iEEG)
        if isinstance(mon, tvb_monitors.EEG):
            params = {}
            sigma = getattr(mon, "sigma", 1.0)
            if sigma is not None:
                params["conductivity"] = tvbo_datamodel.Parameter(
                    name="conductivity", value=float(sigma)
                )
                cr_args.append(tvbo_datamodel.Argument(
                    name="sigma", value=float(sigma),
                ))
            ref = getattr(mon, "reference", None)
            if ref:
                params["reference_electrode"] = tvbo_datamodel.Parameter(
                    name="reference_electrode", value=str(ref)
                )
                cr_args.append(tvbo_datamodel.Argument(
                    name="reference", value=str(ref),
                ))
            if params:
                obs_kwargs["parameters"] = params
            # Store sensor network reference
            sensors = getattr(mon, "sensors", None)
            if sensors is not None:
                sensor_name = getattr(sensors, "title", "eeg_sensors")
                obs_kwargs["data_source"] = tvbo_datamodel.DataSource(
                    name="sensor_network",
                    description=f"EEG sensor array: {sensor_name}",
                )

        if isinstance(mon, tvb_monitors.MEG):
            sensors = getattr(mon, "sensors", None)
            if sensors is not None:
                sensor_name = getattr(sensors, "title", "meg_sensors")
                obs_kwargs["data_source"] = tvbo_datamodel.DataSource(
                    name="sensor_network",
                    description=f"MEG sensor array: {sensor_name}",
                )

        if isinstance(mon, tvb_monitors.iEEG):
            sigma = getattr(mon, "sigma", 1.0)
            if sigma is not None:
                obs_kwargs["parameters"] = {
                    "conductivity": tvbo_datamodel.Parameter(
                        name="conductivity", value=float(sigma)
                    )
                }
                cr_args.append(tvbo_datamodel.Argument(
                    name="sigma", value=float(sigma),
                ))
            sensors = getattr(mon, "sensors", None)
            if sensors is not None:
                sensor_name = getattr(sensors, "title", "seeg_sensors")
                obs_kwargs["data_source"] = tvbo_datamodel.DataSource(
                    name="sensor_network",
                    description=f"SEEG sensor array: {sensor_name}",
                )

        obs_kwargs["class_reference"] = tvbo_datamodel.ClassReference(
            name=tvb_class_name,
            module="tvb.simulator.monitors",
            constructor_args=cr_args,
        )

        observations[obs_name] = tvbo_datamodel.Observation(**obs_kwargs)

    return observations


def to_tvb_monitor(observation):
    """Convert a tvbo Observation to a TVB Monitor instance.

    Uses ``class_reference`` to reconstruct the exact TVB Monitor subclass
    with all constructor arguments including HRF kernel type.

    Parameters
    ----------
    observation : tvbo.datamodel.schema.Observation
        The tvbo observation to convert.

    Returns
    -------
    tvb.simulator.monitors.Monitor
        Configured TVB monitor instance.
    """
    from tvb.simulator import monitors as tvb_monitors
    from tvb.datatypes import equations as tvb_equations

    cr = getattr(observation, 'class_reference', None)
    name = str(observation.name)
    period = observation.period

    # Resolve TVB monitor class from class_reference or name
    _NAME_TO_MONITOR = {
        "Raw": tvb_monitors.Raw,
        "SubSample": tvb_monitors.SubSample,
        "TemporalAverage": tvb_monitors.TemporalAverage,
        "GlobalAverage": tvb_monitors.GlobalAverage,
        "SpatialAverage": tvb_monitors.SpatialAverage,
        "EEG": tvb_monitors.EEG,
        "MEG": tvb_monitors.MEG,
        "iEEG": tvb_monitors.iEEG,
        "Bold": tvb_monitors.Bold,
        "BoldRegionROI": tvb_monitors.BoldRegionROI,
        "AfferentCoupling": tvb_monitors.AfferentCoupling,
        "AfferentCouplingTemporalAverage": tvb_monitors.AfferentCouplingTemporalAverage,
    }

    # Also map tvbo observation names to TVB class names (fallback)
    _OBS_NAME_TO_TVB = {
        "BOLD_TVB": "Bold",
        "BOLD_Gamma": "Bold",
        "BOLD_DoubleExponential": "Bold",
        "BOLD_MixtureOfGammas": "Bold",
        "BOLD_RegionROI": "BoldRegionROI",
    }

    _HRF_CLASSES = {
        "FirstOrderVolterra": tvb_equations.FirstOrderVolterra,
        "Gamma": tvb_equations.Gamma,
        "DoubleExponential": tvb_equations.DoubleExponential,
        "MixtureOfGammas": tvb_equations.MixtureOfGammas,
    }

    if cr is not None:
        tvb_class_name = str(cr.name)
    else:
        tvb_class_name = _OBS_NAME_TO_TVB.get(name, name)

    mon_class = _NAME_TO_MONITOR.get(tvb_class_name)
    if mon_class is None:
        raise ValueError(f"Unknown TVB monitor class: {tvb_class_name}")

    kwargs = {}
    if period is not None and tvb_class_name != "Raw":
        kwargs["period"] = float(period)

    # Process constructor_args from class_reference
    constructor_args = []
    if cr is not None:
        constructor_args = getattr(cr, 'constructor_args', None) or []

    for arg in constructor_args:
        arg_name = str(arg.name)
        if arg_name == "hrf_kernel" and arg.value is not None:
            hrf_cls = _HRF_CLASSES.get(str(arg.value))
            if hrf_cls:
                hrf = hrf_cls()
                # Populate HRF parameters from pipeline
                for step in (observation.pipeline or []):
                    step_name = str(getattr(step, 'name', ''))
                    if step_name in ('hrf_kernel', 'hemodynamic_response'):
                        eq = getattr(step, 'equation', None)
                        if eq:
                            eq_params = getattr(eq, 'parameters', None) or {}
                            for pk, pv in (eq_params.items() if hasattr(eq_params, 'items') else []):
                                val = getattr(pv, 'value', None)
                                if val is not None and str(pk) in hrf.parameters:
                                    hrf.parameters[str(pk)] = float(val)
                kwargs["hrf_kernel"] = hrf
        elif arg_name == "hrf_length" and arg.value is not None:
            kwargs["hrf_length"] = float(arg.value)
        elif arg_name == "sigma" and arg.value is not None:
            kwargs["sigma"] = float(arg.value)
        elif arg_name == "reference" and arg.value is not None:
            kwargs["reference"] = str(arg.value)

    # Load sensors for projection monitors from data_source
    if tvb_class_name in ("EEG", "MEG", "iEEG"):
        _load_projection_monitor_sensors(observation, tvb_class_name, kwargs)

    return mon_class(**kwargs)


def _load_projection_monitor_sensors(observation, tvb_class_name, kwargs):
    """Load sensors and projection for a projection-type monitor.

    Reads the sensor network referenced by ``observation.data_source``
    and populates ``kwargs`` with sensor and projection objects.
    """
    from tvb.datatypes import sensors as tvb_sensors
    from tvb.datatypes import projections as tvb_projections

    _SENSOR_TYPES = {
        "EEG": tvb_sensors.SensorsEEG,
        "MEG": tvb_sensors.SensorsMEG,
        "iEEG": tvb_sensors.SensorsInternal,
    }
    _PROJ_TYPES = {
        "EEG": tvb_projections.ProjectionSurfaceEEG,
        "MEG": tvb_projections.ProjectionSurfaceMEG,
        "iEEG": tvb_projections.ProjectionSurfaceSEEG,
    }

    ds = getattr(observation, 'data_source', None)
    if ds is None:
        return

    # Try to load sensor network from path
    sensor_path = getattr(ds, 'path', None)
    if not sensor_path:
        return

    try:
        from tvbo import Network
        import numpy as np
        import os

        # Resolve path relative to database/networks if not absolute
        if not os.path.isabs(sensor_path):
            db_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "database", "networks"
            )
            sensor_path = os.path.join(db_dir, sensor_path)

        net = Network.from_file(sensor_path)
        sensor_cls = _SENSOR_TYPES.get(tvb_class_name)
        if sensor_cls is None:
            return

        # Build sensor locations from node positions
        locations = np.array([
            [n.position.x, n.position.y, n.position.z]
            for n in net.nodes
        ])
        labels = np.array([n.label for n in net.nodes])

        sensors = sensor_cls(locations=locations, labels=labels)

        # MEG sensors also need orientations
        if tvb_class_name == "MEG":
            orientations = []
            for n in net.nodes:
                params = getattr(n, 'parameters', None) or {}
                ox = float(getattr(params.get('orientation_x'), 'value', 0))
                oy = float(getattr(params.get('orientation_y'), 'value', 0))
                oz = float(getattr(params.get('orientation_z'), 'value', 0))
                orientations.append([ox, oy, oz])
            sensors.orientations = np.array(orientations)

        kwargs["sensors"] = sensors

        # Load projection from gain edge parameter
        gain_data = net.matrix("gain", format="dense")
        if gain_data is not None:
            proj_cls = _PROJ_TYPES.get(tvb_class_name)
            if proj_cls is not None:
                kwargs["projection"] = proj_cls(projection_data=gain_data)
    except Exception:
        pass  # Sensors optional — analytic fallback available


def _extract_environment() -> tvbo_datamodel.SoftwareEnvironment | None:
    """Create SoftwareEnvironment metadata for TVB."""
    import platform as _platform

    try:
        import tvb as _tvb
        tvb_version = getattr(_tvb, "__version__", "unknown")
    except ImportError:
        tvb_version = "unknown"

    return tvbo_datamodel.SoftwareEnvironment(
        name="TVB",
        version=str(tvb_version),
        platform=_platform.platform(),
    )


def from_tvb_zip(zip_path):
    """Import a TVB connectivity ZIP into a tvbo Network.

    TVB ZIPs contain: ``weights.txt``, ``tract_lengths.txt``,
    ``centres.txt``.

    Parameters
    ----------
    zip_path : str or Path
        Path to TVB connectivity ZIP file.

    Returns
    -------
    Network
        Network instance with loaded arrays ready for ``save()``.
    """
    import zipfile
    import io
    from tvbo.classes.network import Network

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        weights = np.loadtxt(io.BytesIO(zf.read("weights.txt")))
        lengths = np.loadtxt(io.BytesIO(zf.read("tract_lengths.txt")))
        centres_raw = zf.read("centres.txt").decode()

    labels, coords = [], []
    for line in centres_raw.strip().split("\n"):
        parts = line.split()
        labels.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])

    n = weights.shape[0]
    nodes = [
        tvbo_datamodel.Node(
            id=i, label=labels[i],
            position=tvbo_datamodel.Coordinate(
                x=float(coords[i][0]),
                y=float(coords[i][1]),
                z=float(coords[i][2]),
            ),
        )
        for i in range(n)
    ]
    net = Network(nodes=nodes, edges=[], number_of_nodes=n)
    net.set_matrix("weight", weights)
    net.set_matrix("length", lengths)
    net.label = zip_path.stem
    net.descriptor = "SC"
    return net


def from_tvb(connectivity):
    """Import a live TVB Connectivity object into a tvbo Network.

    Lossless conversion — all TVB fields are preserved as tvbo Node
    parameters and Network-level metadata.

    Parameters
    ----------
    connectivity : tvb.datatypes.connectivity.Connectivity
        A configured TVB Connectivity instance.

    Returns
    -------
    Network
        Network instance with arrays loaded, ready for ``save()``.
    """
    from tvbo.classes.network import Network

    conn = connectivity
    weights = np.asarray(conn.weights, dtype="float64")
    lengths = np.asarray(conn.tract_lengths, dtype="float64")
    centres = np.asarray(conn.centres, dtype="float64")
    labels = list(conn.region_labels)
    n = weights.shape[0]

    nodes = []
    for i in range(n):
        node = tvbo_datamodel.Node(
            id=i,
            label=str(labels[i]),
            position=tvbo_datamodel.Coordinate(
                x=float(centres[i, 0]),
                y=float(centres[i, 1]),
                z=float(centres[i, 2]),
            ),
        )
        if conn.cortical is not None and len(conn.cortical) == n:
            node.parameters["cortical"] = tvbo_datamodel.Parameter(
                name="cortical", value=float(conn.cortical[i]),
            )
        if conn.areas is not None and len(conn.areas) == n:
            node.parameters["area"] = tvbo_datamodel.Parameter(
                name="area", value=float(conn.areas[i]),
            )
        if conn.hemispheres is not None and len(conn.hemispheres) == n:
            node.parameters["hemisphere"] = tvbo_datamodel.Parameter(
                name="hemisphere", value=float(conn.hemispheres[i]),
            )
        nodes.append(node)

    net = Network(nodes=nodes, edges=[], number_of_nodes=n)
    net.set_matrix("weight", weights)
    net.set_matrix("length", lengths)
    net.label = getattr(conn, "title", None) or "TVB Connectivity"
    net.descriptor = "SC"

    if conn.orientations is not None and len(conn.orientations) == n:
        object.__setattr__(net, '_orientations', np.asarray(conn.orientations))

    speed = np.asarray(conn.speed).ravel()
    cs_val = float(speed[0]) if len(speed) > 0 else 3.0
    net.parameters["conduction_speed"] = tvbo_datamodel.Parameter(
        name="conduction_speed", label="v", value=cs_val, unit="mm_per_ms",
    )

    return net


def from_tvb_surface(connectivity, surface, region_mapping):
    """Create a multi-level tvbo Network from TVB surface simulation data.

    Produces two linked networks:

    1. **Region-level** (parent): from TVB Connectivity
    2. **Vertex-level** (child): mesh + region_mapping linking vertices
       to regions via hierarchical ``node_mapping``

    Parameters
    ----------
    connectivity : tvb.datatypes.connectivity.Connectivity
        Configured TVB Connectivity (region-level).
    surface : tvb.datatypes.surfaces.Surface
        TVB CorticalSurface (or any Surface subclass).
    region_mapping : tvb.datatypes.region_mapping.RegionMapping
        TVB RegionMapping mapping vertices to regions.

    Returns
    -------
    tuple[Network, Network]
        ``(region_network, surface_network)``
    """
    from tvbo.classes.network import Network

    region_net = from_tvb(connectivity)

    vertices = np.asarray(surface.vertices, dtype="float32")
    triangles = np.asarray(surface.triangles, dtype="int32")
    normals = np.asarray(surface.vertex_normals, dtype="float32")
    mapping = np.asarray(region_mapping.array_data, dtype="int32")
    n_vertices = vertices.shape[0]
    n_elements = triangles.shape[0]

    surface_nodes = [
        tvbo_datamodel.Node(
            id=i, label=f"vertex_{i}",
            position=tvbo_datamodel.Coordinate(
                x=float(vertices[i, 0]),
                y=float(vertices[i, 1]),
                z=float(vertices[i, 2]),
            ),
        )
        for i in range(n_vertices)
    ]

    surface_net = Network(
        nodes=surface_nodes, edges=[], number_of_nodes=n_vertices,
    )
    surface_net.label = f"Surface ({n_vertices} vertices, {n_elements} triangles)"
    surface_net.descriptor = "surface"

    mesh = tvbo_datamodel.Mesh(
        label=getattr(surface, "surface_type", "CorticalSurface"),
        element_type="triangle",
        number_of_vertices=n_vertices,
        number_of_elements=n_elements,
    )
    object.__setattr__(surface_net, '_mesh', mesh)
    object.__setattr__(surface_net, '_mesh_vertices', vertices)
    object.__setattr__(surface_net, '_mesh_elements', triangles)
    object.__setattr__(surface_net, '_mesh_normals', normals)

    surface_net.set_node_mapping(
        mapping,
        parent_network=region_net,
        dataset_path="/mesh/region_mapping",
    )

    return region_net, surface_net


def to_tvb(network):
    """Export a tvbo Network to a TVB Connectivity object.

    Parameters
    ----------
    network : Network
        tvbo Network instance with weights and lengths matrices.

    Returns
    -------
    tvb.datatypes.connectivity.Connectivity
        Configured TVB Connectivity ready for simulation.
    """
    from tvb.datatypes.connectivity import Connectivity

    _weights = np.asarray(network.weights_matrix, dtype=float)
    _lengths = np.asarray(network.lengths_matrix, dtype=float)
    _centres = np.asarray(
        list(network.get_centers().values()), dtype=float
    )
    cs_param = getattr(network, "conduction_speed", None)
    cs_value = (
        cs_param.value
        if cs_param and hasattr(cs_param, "value")
        else 3.0
    )
    _speed = np.asarray([cs_value], dtype=float)
    tvb_conn = Connectivity(
        weights=_weights,
        tract_lengths=_lengths,
        centres=_centres,
        region_labels=network.atlas.region_labels,
        speed=_speed,
    )
    tvb_conn.configure()
    return tvb_conn


# ── Simulator-level conversion ────────────────────────────────────

def from_tvb_simulator(
    sim: Any,
    experiment_id: Union[int, None] = None,
) -> tvbo_datamodel.SimulationExperiment:
    """Convert a TVB Simulator into a tvbo SimulationExperiment datamodel.

    Parameters
    ----------
    sim : tvb.simulator.simulator.Simulator
        A configured TVB Simulator instance.
    experiment_id : int, optional
        Experiment identifier. Defaults to the simulator's GID.

    Returns
    -------
    tvbo_datamodel.SimulationExperiment
        Schema-valid datamodel instance.
    """
    if not experiment_id:
        try:
            experiment_id = sim.gid.int
        except Exception:
            experiment_id = 1

    dynamics = _extract_dynamics(sim)
    network = from_tvb(sim.connectivity)

    exp = tvbo_datamodel.SimulationExperiment(
        id=experiment_id,
        model=dynamics.name,
        dynamics=dynamics,
        coupling=_extract_coupling(sim),
        integration=_extract_integration(sim, dynamics),
        network=network,
        stimulation=_extract_stimulus(sim),
        environment=_extract_environment(),
        observations=_extract_observations(sim),
    )

    return exp
