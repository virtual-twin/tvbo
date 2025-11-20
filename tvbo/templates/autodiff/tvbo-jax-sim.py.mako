# -*- coding: utf-8 -*-
<%
    from tvbo.export.code import render_expression
    jaxcode = lambda expr: render_expression(expr, format='jax')
    import numpy as np

    integration = experiment.integration if hasattr(experiment, 'integration') else None
    # Per-state sigma from experiment as a fallback; runtime can override via state.noise.sigma_vec
    sw = np.asarray(experiment.noise_sigma_array).reshape((-1,))
    # Always run the stochastic path; when sigma is zero, noise will be zero.
    # This enables vmapping over sigma by passing it through SimulationState.noise.sigma_vec.
    stochastic = True
    import builtins as _builtins
    any_delays = experiment.horizon > 1
    # Obtain monitors regardless of whether experiment.monitors is a field (dict/list) or a method
    _mon = getattr(experiment, 'monitors', None)
    try:
        monitors_seq = experiment.monitors() if callable(_mon) else (_mon.values() if hasattr(_mon, 'values') else (_mon or []))
    except TypeError:
        monitors_seq = _mon.values() if hasattr(_mon, 'values') else (_mon or [])
    model = experiment.local_dynamics
    coupling = experiment.coupling
    integration = experiment.integration

    dt = integration.step_size if integration is not None else 0.1
    cvar = [i for i, sv in enumerate(experiment.local_dynamics.state_variables.values()) if sv.coupling_variable]
    vois = [sv.name for sv in experiment.local_dynamics.state_variables.values() if sv.variable_of_interest]

    svars = list(model.state_variables.keys())
    svars_is_vois = svars == vois

    ## print('simulation delayed:', any_delays)
%>

import jax
from tvbo.data.types import TimeSeries
## Usefull shorthands
import jax.numpy as jnp

<%include file="/tvbo-jax-coupling.py.mako" />
<%include file="/tvbo-jax-dfuns.py.mako" />
<%include file="/tvbo-jax-integrate.py.mako" />
## <%include file="/jax-monitors.py.mako" />
<%include file="/jax-noise.py.mako" />
<%namespace name="monitors" file="/jax-monitors.py.mako"/>
<%namespace name="utils" file="/jax-utils.py.mako"/>

## Monitors
## % for i, monitor in enumerate(monitors_seq):
## ${monitors.create_monitor(i, monitor, dt)}
## % endfor
def monitor_raw(time_steps, trace, params, t_offset = 0):
    dt = ${dt}
    return TimeSeries(time=(time_steps + t_offset) * dt, data=trace, title = "Raw")

## Transformation for derived parameters
def transform_parameters(_p):
    ${", ".join([p.name for p in experiment.local_dynamics.parameters.values()])} = _p.${", _p.".join([p.name for p in experiment.local_dynamics.parameters.values()])}
    \
## % for par in [p.name for p in experiment.local_dynamics.parameters.values()]:
## ${par}, \
## % endfor
## = params_dfun

    % for par in experiment.local_dynamics.derived_parameters.values():
    ${par.name} = ${par.equation.rhs}
    % endfor
    %if len(experiment.local_dynamics.derived_parameters.values()) > 0:
    _p.${", _p.".join([p.name for p in experiment.local_dynamics.derived_parameters.values()])} = ${", ".join([p.name for p in experiment.local_dynamics.derived_parameters.values()])}
    % endif
    return _p

##     return (\
## % for par in [p.name for p in experiment.local_dynamics.parameters.values()] + [p.name for p in experiment.local_dynamics.derived_parameters.values()]:
## ${par}, \
## % endfor
## )
c_vars = ${utils.array_input(np.array(cvar))}

## Main Function
## def kernel(initial_conditions, weights, delay_indices, dt, nt, noise, params_integrate, params_monitors):
## TODO: provide def kernel_n(state, noise) for explicitly providing noise, ...
def kernel(state):
    # problem dimensions
    n_nodes = ${experiment.network.number_of_regions}
    n_svar = ${len(experiment.local_dynamics.state_variables)}
    n_cvar = ${len(cvar)}
    n_modes = ${experiment.local_dynamics.number_of_modes}
    nh = ${experiment.horizon}

    %if any_delays:
    current_state, history = (state.initial_conditions.data[-1], state.initial_conditions.data[-nh:, c_vars].transpose(1, 0, 2, 3))
    % if small_dt:
        history = jnp.concatenate([jnp.empty((n_cvar, state.nt, n_nodes, n_modes)), history], axis = 1)
    %endif
    %else:
    current_state, history = (state.initial_conditions.data[-1], None) ## history = current_state
    %endif

    ics = (history, current_state)
    weights = state.network.weights_matrix

    dn = jnp.arange(n_nodes) * jnp.ones((n_nodes, n_nodes)).astype(int)
    idelays = jnp.round(state.network.lengths_matrix / state.network.conduction_speed.value / state.dt).astype(int)
    di = -1 * idelays -1
    delay_indices = (di, dn)

    dt = state.dt
    nt = state.nt
    time_steps = jnp.arange(0, nt)

    # Generate batch noise using xi with per-state sigma_vec.
    # Prefer state-provided sigma_vec (supports vmapped sweeps); fallback to experiment-level constants.
    seed = getattr(state.noise, 'seed', 0) if hasattr(state.noise, 'seed') else 0
    try:
        sigma_vec_runtime = getattr(state.noise, 'sigma_vec', None)
    except Exception:
        sigma_vec_runtime = None
    sigma_vec = sigma_vec_runtime if sigma_vec_runtime is not None else ${utils.array_input(sw)}
    noise = g(dt, nt, n_svar, n_nodes, n_modes, seed=seed, sigma_vec=sigma_vec)

    ## initial conditions go through the carry of scan

    p = transform_parameters(state.parameters.local_dynamics)
    params_integrate = (p, state.parameters.coupling, state.stimulus)

    op = lambda ics, external_input: integrate(ics, weights, dt, params_integrate, delay_indices, external_input)
    latest_carry, res = jax.lax.scan(op, ics, (time_steps, noise))

    ## Extract trace and node_coupling if present
    trace = res

    ## Extract new initial conditions if needed
    % if return_new_ics:
    ## valid if nt > nh - does not error
    new_ics = TimeSeries(time = jnp.linspace(nt-nh+1, nt, nh) * dt, data = trace[-nh:, :, :, :], title = "New initial conditions")
    ## ics = namedtuple("initial_conditions", ["current_state", "history"])
    ## % if not small_dt:
    ## new_ics = ics(latest_carry[1], latest_carry[0])
    ## % else:
    ## new_current_state = trace[-1, :, :, :]
    ## % if any_delays:
    ## cvar = ${utils.array_input(np.array(cvar))}
    ## new_history = jnp.transpose(trace[-nh:, cvar, :, :], (1, 0, 2, 3))
    ## % else:
    ## new_history = trace[-1, :, :, :]
    ## % endif
    ## new_ics = ics(new_current_state, new_history)
    ## % endif
    % endif

    ## Apply variables of interest and generate derived variables


    ## Apply variables of interest and generate derived variables
    <%
    vois = [sv.name for sv in experiment.local_dynamics.state_variables.values() if sv.variable_of_interest]
    svars = [sv.name for sv in experiment.local_dynamics.state_variables.values()]
    svars_is_vois = svars == vois
    has_output_transforms = len(experiment.local_dynamics.output_transforms) > 0
    %>
    ## Generate expressions for derived variables and potentially remove and reorder state variables
    % if not svars_is_vois or has_output_transforms:
    % if has_output_transforms:
##     \
## % for par in [p.name for p in experiment.local_dynamics.parameters.values()] + [p.name for p in experiment.local_dynamics.derived_parameters.values()]:
## ${par}, \
##     % endfor
## = params_integrate[0]
    ${", ".join([p.name for p in experiment.local_dynamics.parameters.values()] + [p.name for p in experiment.local_dynamics.derived_parameters.values()])} = p.${", p.".join([p.name for p in experiment.local_dynamics.parameters.values()] + [p.name for p in experiment.local_dynamics.derived_parameters.values()])}
    % endif

    # state variables: ${svars} to variables of interest: ${vois}
    % for var in vois:
        % if var in svars:
    ${var} = trace[:, [${(svars.index(var))}], :]
        % else:
    ${var} = ${utils.generate_derived_expression(var, svars)}
        % endif
    % endfor
    % for trafo in experiment.local_dynamics.output_transforms.values():
    ${trafo.name} = ${jaxcode(trafo.equation.rhs)}
    % endfor

    trace = jnp.hstack((
        % for var in vois:
            ${var},
        % endfor
        % for trafo in experiment.local_dynamics.output_transforms.values():
            ${trafo.name},
        % endfor
        ))
    % endif

    t_offset = ${experiment.integration.current_step}
    time_steps = time_steps + 1

    ## Apply monitors to trace
    ## params_monitors = state.monitor_parameters
    ## ${('result = [' if (len(experiment.monitors()) > 1) else 'result = ')} \
    ## % for i, monitor in enumerate(experiment.monitors()):
    ## ${monitors.apply_monitor(i, monitor)}
    ## % endfor
    ## ${']' if (len(experiment.monitors()) > 1) else ''}
    ## ${'result = [result[0]]' if (len(experiment.monitors()) == 1) else ''}
    % if not return_new_ics:
    ## Build labels for TimeSeries so indexing and plotting work robustly
    labels_dimensions = {
        "Time": None,
        "State Variable": ${(vois + [trafo.name for trafo in experiment.local_dynamics.output_transforms.values()])},
        "Space": ${list(experiment.network.parcellation.region_labels) if getattr(experiment.network.parcellation, 'region_labels', None) else [str(i) for i in range(experiment.network.number_of_regions)]},
        "Mode": ${[f"m{i}" for i in range(experiment.local_dynamics.number_of_modes)]},
    }
    return TimeSeries(time=(time_steps + t_offset) * dt, data=trace, title = "Raw", sample_period=dt, labels_dimensions=labels_dimensions)
    % else:
    return (result, new_ics)
    % endif
