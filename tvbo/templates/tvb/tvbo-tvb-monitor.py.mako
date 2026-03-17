<%
# Generate TVB monitor instances from observation metadata.
# Each Observation carries a class_reference that maps to the exact TVB Monitor class.
# For projection monitors (EEG, MEG, iEEG), sensor data is loaded from the
# sensor network referenced by observation.data_source.
observations = getattr(experiment, 'observations', None) or {}

# HRF kernel class names that need import from tvb.datatypes.equations
_HRF_CLASSES = {
    'FirstOrderVolterra', 'Gamma', 'DoubleExponential', 'MixtureOfGammas',
}

# Projection monitor classes that need sensor data
_PROJECTION_CLASSES = {'EEG', 'MEG', 'iEEG'}

# TVB sensor class per monitor type
_SENSOR_CLASSES = {
    'EEG': ('SensorsEEG', 'tvb.datatypes.sensors'),
    'MEG': ('SensorsMEG', 'tvb.datatypes.sensors'),
    'iEEG': ('SensorsInternal', 'tvb.datatypes.sensors'),
}

# TVB projection class per monitor type
_PROJ_CLASSES = {
    'EEG': ('ProjectionSurfaceEEG', 'tvb.datatypes.projections'),
    'MEG': ('ProjectionSurfaceMEG', 'tvb.datatypes.projections'),
    'iEEG': ('ProjectionSurfaceSEEG', 'tvb.datatypes.projections'),
}

# Collect imports, sensor setup code, and monitor expressions
monitor_imports = set()
monitor_imports.add('from tvb.simulator.monitors import *')
hrf_imports_needed = set()
sensor_imports_needed = set()  # (module, class_name)
projection_imports_needed = set()
sensor_setup_lines = []  # Code lines for sensor/projection loading
monitor_exprs = []

for obs_name, obs in (observations.items() if hasattr(observations, 'items') else []):
    cr = getattr(obs, 'class_reference', None)
    if cr is None:
        continue

    tvb_class = str(cr.name)
    period = getattr(obs, 'period', None)

    # Build constructor kwargs
    kwargs_parts = []

    # Period (skip for Raw and AfferentCoupling — they record at every step)
    if period is not None and tvb_class not in ('Raw', 'AfferentCoupling'):
        kwargs_parts.append(f'period={period}')

    # --- HRF kernel for BOLD monitors ---
    constructor_args = getattr(cr, 'constructor_args', None) or []
    for arg in constructor_args:
        arg_name = str(arg.name)
        if arg_name == 'hrf_kernel' and arg.value is not None:
            hrf_kernel_name = str(arg.value)
            if hrf_kernel_name in _HRF_CLASSES:
                hrf_imports_needed.add(hrf_kernel_name)
                # Collect HRF parameters from pipeline
                hrf_params = {}
                pipeline = getattr(obs, 'pipeline', None) or []
                for step in pipeline:
                    step_name = str(getattr(step, 'name', ''))
                    if step_name in ('hemodynamic_response', 'hrf_kernel',
                                     'HemodynamicResponseFunctionTVB'):
                        eq = getattr(step, 'equation', None)
                        if eq:
                            eq_params = getattr(eq, 'parameters', None) or {}
                            for pk, pv in (eq_params.items() if hasattr(eq_params, 'items') else []):
                                val = getattr(pv, 'value', None)
                                if val is not None:
                                    hrf_params[str(pk)] = val
                if hrf_params:
                    params_str = ', '.join(f'"{k}": {v}' for k, v in hrf_params.items())
                    kwargs_parts.append(f'hrf_kernel={hrf_kernel_name}(parameters={{{params_str}}})')
                else:
                    kwargs_parts.append(f'hrf_kernel={hrf_kernel_name}()')
        elif arg_name == 'hrf_length' and arg.value is not None:
            kwargs_parts.append(f'hrf_length={arg.value}')

    # --- Projection monitor: sigma/reference from constructor_args ---
    if tvb_class in _PROJECTION_CLASSES:
        for arg in constructor_args:
            arg_name = str(arg.name)
            if arg_name == 'sigma' and arg.value is not None:
                kwargs_parts.append(f'sigma={arg.value}')
            elif arg_name == 'reference' and arg.value is not None:
                kwargs_parts.append(f'reference="{arg.value}"')

    # --- Projection monitor: also check observation.parameters for sigma/reference ---
    obs_params = getattr(obs, 'parameters', None) or {}
    if tvb_class in ('EEG', 'iEEG'):
        # Check generic parameter names (conductivity) and TVB names (sigma)
        for generic_name, tvb_name in [('conductivity', 'sigma'), ('sigma', 'sigma')]:
            pv = obs_params.get(generic_name) if hasattr(obs_params, 'get') else None
            if pv is not None:
                val = getattr(pv, 'value', None)
                if val is not None and f'sigma=' not in ','.join(kwargs_parts):
                    kwargs_parts.append(f'sigma={val}')
    if tvb_class == 'EEG':
        for ref_name in ('reference_electrode', 'reference'):
            pv = obs_params.get(ref_name) if hasattr(obs_params, 'get') else None
            if pv is not None:
                val = getattr(pv, 'value', None)
                if val is not None and 'reference=' not in ','.join(kwargs_parts):
                    kwargs_parts.append(f'reference="{val}"')

    # --- Projection monitor: sensor and projection loading ---
    if tvb_class in _PROJECTION_CLASSES:
        ds = getattr(obs, 'data_source', None)
        if ds is not None:
            sensor_info = _SENSOR_CLASSES[tvb_class]
            sensor_imports_needed.add(sensor_info)
            sensor_var = f'sensors_{obs_name.lower()}'

            ds_path = getattr(ds, 'path', None)
            if ds_path:
                # Generate sensor loading from file using TVB's from_file
                # Extract just the filename stem for from_file
                import os
                sensor_fname = os.path.splitext(os.path.basename(ds_path))[0]
                # TVB's from_file expects .txt files
                sensor_setup_lines.append(
                    f'{sensor_var} = {sensor_info[0]}.from_file("{sensor_fname}.txt")'
                )
            else:
                sensor_setup_lines.append(
                    f'{sensor_var} = {sensor_info[0]}.from_file()'
                )

            kwargs_parts.append(f'sensors={sensor_var}')

            # Check for projection: only if we have a sensor network with data_file
            proj_info = _PROJ_CLASSES[tvb_class]
            projection_imports_needed.add(proj_info)
            proj_var = f'projection_{obs_name.lower()}'
            # Use TVB's default projection for a from_file pattern
            sensor_setup_lines.append(
                f'try:\n'
                f'    {proj_var} = {proj_info[0]}.from_file()\n'
                f'except Exception:\n'
                f'    {proj_var} = None  # analytic fallback'
            )
            kwargs_parts.append(f'projection={proj_var}')

    kwargs_str = ', '.join(kwargs_parts)
    monitor_exprs.append(f'{tvb_class}({kwargs_str})')

# Default to Raw() if no observations with class_reference
if not monitor_exprs:
    monitor_exprs = ['Raw()']

# Build additional imports
if hrf_imports_needed:
    monitor_imports.add('from tvb.datatypes.equations import ' + ', '.join(sorted(hrf_imports_needed)))
for cls_name, module in sensor_imports_needed:
    monitor_imports.add(f'from {module} import {cls_name}')
for cls_name, module in projection_imports_needed:
    monitor_imports.add(f'from {module} import {cls_name}')
%>
##
% for imp in sorted(monitor_imports):
${imp}
% endfor
% if sensor_setup_lines:

# Load sensor and projection data for projection monitors
% for line in sensor_setup_lines:
${line}
% endfor
% endif

monitors = [${', '.join(monitor_exprs)}]

