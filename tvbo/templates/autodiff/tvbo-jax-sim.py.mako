# -*- coding: utf-8 -*-
<%
    from tvbo.export.code import render_expression
    jaxcode = lambda expr: render_expression(expr, format='jax')
    # For rendering model objects (DerivedParameter, DerivedVariable, etc.)
    jaxcode_obj = lambda obj: experiment.local_dynamics.render_equation(obj, format='jax')
    import numpy as np
    import networkx as nx
    from sympy import Symbol

    integration = experiment.integration if hasattr(experiment, 'integration') else None
    # Per-state sigma from experiment as a fallback; runtime can override via state.noise.sigma_vec
    sw = np.asarray(experiment.noise_sigma_array).reshape((-1,))
    # Always run the stochastic path; when sigma is zero, noise will be zero.
    # This enables vmapping over sigma by passing it through SimulationState.noise.sigma_vec.
    stochastic = True
    import builtins as _builtins
    is_delayed = experiment.horizon > 1
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
    
    # Identify output derived variables that depend on dfun arguments (coupling, t, noise, stimulus)
    # These must be computed inside integrate and returned as part of scan output
    dfun_args = set(model.coupling_terms.keys()) | {'t', 'noise', 'stimulus'}
    derived_var_names = list(model.derived_variables.keys())
    derived_param_names = list(model.derived_parameters.keys())
    param_names = [p.name for p in model.parameters.values()]
    output_vars = list(model.output) if model.output else svars
    has_output = len(output_vars) > 0
    
    # Get dependency graph
    G = model.get_dependency_tree()
    
    # Categorize output derived vars: those needing integrate vs those computable from trace
    output_derived_in_integrate = []  # Must be computed in integrate (depend on coupling/t/noise)
    output_derived_from_trace = []    # Can be computed post-hoc from trace
    
    for var_name in output_vars:
        if var_name in derived_var_names:
            var_sym = Symbol(var_name)
            if var_sym in G.nodes():
                ancestors = nx.ancestors(G, var_sym)
                dfun_deps = [str(a) for a in ancestors if str(a) in dfun_args]
                if dfun_deps:
                    output_derived_in_integrate.append(var_name)
                else:
                    output_derived_from_trace.append(var_name)
            else:
                output_derived_from_trace.append(var_name)
    
    # Collect ALL derived variables needed in integrate (in dependency order)
    all_integrate_derived_vars = []
    for var_name in output_derived_in_integrate:
        var_sym = Symbol(var_name)
        if var_sym in G.nodes():
            ancestors = nx.ancestors(G, var_sym)
            for anc in ancestors:
                anc_name = str(anc)
                if anc_name in derived_var_names and anc_name not in all_integrate_derived_vars:
                    all_integrate_derived_vars.append(anc_name)
            if var_name not in all_integrate_derived_vars:
                all_integrate_derived_vars.append(var_name)
    
    # Sort by topological order
    if all_integrate_derived_vars:
        all_syms = [Symbol(n) for n in all_integrate_derived_vars]
        subG = G.subgraph([s for s in all_syms if s in G.nodes()])
        try:
            topo_order = list(nx.topological_sort(subG))
            all_integrate_derived_vars = [str(s) for s in topo_order if str(s) in all_integrate_derived_vars]
        except:
            pass

    ## print('simulation delayed:', is_delayed)
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
<%namespace name="obs" file="/jax-observation.py.mako"/>

## Monitors
## % for i, monitor in enumerate(monitors_seq):
## ${monitors.create_monitor(i, monitor, dt)}
## % endfor
def monitor_raw(time_steps, trace, params, t_offset = 0):
    dt = ${dt}
    return TimeSeries(time=(time_steps + t_offset) * dt, data=trace, title = "Raw")

## Observations
% if hasattr(experiment, 'observations') and experiment.observations:
${obs.create_all_observations(experiment)}
% endif

## Transformation for derived parameters
def transform_parameters(_p):
    ${", ".join([p.name for p in experiment.local_dynamics.parameters.values()])} = _p.${", _p.".join([p.name for p in experiment.local_dynamics.parameters.values()])}
    \
## % for par in [p.name for p in experiment.local_dynamics.parameters.values()]:
## ${par}, \
## % endfor
## = params_dfun

    % for par in experiment.local_dynamics.derived_parameters.values():
    ${par.name} = ${jaxcode_obj(par)}
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
c_vars = ${utils.array_input(np.array(cvar))}.astype(jnp.int32)

## Main Function
## def kernel(initial_conditions, weights, delay_indices, dt, nt, noise, params_integrate, params_monitors):
## TODO: provide def kernel_n(state, noise) for explicitly providing noise, ...
def kernel(state):
    # problem dimensions
    n_nodes = ${experiment.network.number_of_regions}
    n_svar = ${len(experiment.local_dynamics.state_variables)}
    n_cvar = ${len(cvar) if len(cvar) > 0 else len(experiment.local_dynamics.state_variables)}
    n_modes = ${experiment.local_dynamics.number_of_modes}
    nh = ${experiment.horizon}

    %if is_delayed:
    %if len(cvar) > 0:
    current_state, history = (state.initial_conditions.data[-1], state.initial_conditions.data[-nh:, c_vars].transpose(1, 0, 2, 3))
    %else:
    ## When no coupling variables are defined, use all state variables for history
    current_state, history = (state.initial_conditions.data[-1], state.initial_conditions.data[-nh:, :].transpose(1, 0, 2, 3))
    %endif
    % if small_dt:
        history = jnp.concatenate([jnp.empty((n_cvar, state.nt, n_nodes, n_modes)), history], axis = 1)
    %endif
    %else:
    current_state, history = (state.initial_conditions.data[-1], None) ## history = current_state
    %endif

    ics = (history, current_state)
    weights = state.network.weights_matrix

    dn = jnp.arange(int(n_nodes)) * jnp.ones((int(n_nodes), int(n_nodes))).astype(jnp.int32)
    idelays = jnp.round(state.network.lengths_matrix / state.network.conduction_speed.value / state.dt).astype(jnp.int32)
    di = -1 * idelays - 1
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

    ## Extract trace and derived vars from scan output
    % if output_derived_in_integrate:
    ## Integrate returned (next_state, _output_derived) or (next_state, cX, _output_derived)
        % if monitor_node_coupling:
    trace, node_coupling, output_derived = res
        % else:
    trace, output_derived = res
        % endif
    % else:
    trace = res
    % endif

    ## Extract new initial conditions if needed
    % if return_new_ics:
    ## valid if nt > nh - does not error
    new_ics = TimeSeries(time = jnp.linspace(nt-nh+1, nt, nh) * dt, data = trace[-nh:, :, :, :], title = "New initial conditions")
    ## ics = namedtuple("initial_conditions", ["current_state", "history"])
    ## % if not small_dt:
    ## new_ics = ics(latest_carry[1], latest_carry[0])
    ## % else:
    ## new_current_state = trace[-1, :, :, :]
    ## % if is_delayed:
    ## cvar = ${utils.array_input(np.array(cvar))}
    ## new_history = jnp.transpose(trace[-nh:, cvar, :, :], (1, 0, 2, 3))
    ## % else:
    ## new_history = trace[-1, :, :, :]
    ## % endif
    ## new_ics = ics(new_current_state, new_history)
    ## % endif
    % endif

    ## Apply output transformations - output is the declarative specification
    <%
    # svars and output_vars already computed at top of template
    # output_derived_in_integrate and output_derived_from_trace also already computed
    %>
    % if has_output:
    % if output_derived_from_trace or output_derived_in_integrate:
    ## Need parameters for derived variable computations
    ${", ".join(param_names + derived_param_names)} = p.${", p.".join(param_names + derived_param_names)}
    % endif

    % if output_derived_from_trace:
    ## Extract state variables from trace for computing derived vars that don't need coupling
    % for i, svar in enumerate(svars):
    ${svar} = trace[:, [${i}], :]
    % endfor

    ## Compute derived vars that only depend on state vars (can be done from trace)
    <%
    # Get dependency-ordered list of needed derived vars for output_derived_from_trace
    trace_derived_deps = []
    for var_name in output_derived_from_trace:
        var_sym = Symbol(var_name)
        if var_sym in G.nodes():
            ancestors = nx.ancestors(G, var_sym)
            for anc in ancestors:
                anc_name = str(anc)
                if anc_name in derived_var_names and anc_name not in trace_derived_deps:
                    trace_derived_deps.append(anc_name)
            if var_name not in trace_derived_deps:
                trace_derived_deps.append(var_name)
    # Sort by topo order
    if trace_derived_deps:
        all_syms = [Symbol(n) for n in trace_derived_deps]
        subG = G.subgraph([s for s in all_syms if s in G.nodes()])
        try:
            topo_order = list(nx.topological_sort(subG))
            trace_derived_deps = [str(s) for s in topo_order if str(s) in trace_derived_deps]
        except:
            pass
    %>
    % for dv_name in trace_derived_deps:
    <%
        dv = experiment.local_dynamics.derived_variables[dv_name]
        rendered_eq = jaxcode_obj(dv)
    %>
    ${dv_name} = ${rendered_eq}
    % endfor
    % endif

    % if output_derived_in_integrate:
    ## Extract output derived vars computed in integrate (from scan output)
    % for i, var_name in enumerate(output_derived_in_integrate):
    ${var_name} = output_derived[:, ${i}:${i+1}, :]
    % endfor
    % endif

    ## Build final output trace
    trace = jnp.hstack((
        % for var_name in output_vars:
            % if var_name in svars:
        trace[:, [${svars.index(var_name)}], :],
            % else:
        ${var_name},
            % endif
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
    <%
    # Use output if specified, otherwise fall back to all state variables
    output_labels = list(experiment.local_dynamics.output) if experiment.local_dynamics.output else [sv.name for sv in experiment.local_dynamics.state_variables.values()]
    %>
    labels_dimensions = {
        "Time": None,
        "State Variable": ${output_labels},
        "Space": ${list(experiment.network.parcellation.region_labels) if getattr(experiment.network.parcellation, 'region_labels', None) else [str(i) for i in range(experiment.network.number_of_regions)]},
        "Mode": ${[f"m{i}" for i in range(experiment.local_dynamics.number_of_modes)]},
    }
    return TimeSeries(time=(time_steps + t_offset) * dt, data=trace, title = "Raw", sample_period=dt, labels_dimensions=labels_dimensions)
    % else:
    return (result, new_ics)
    % endif
