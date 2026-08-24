# -*- coding: utf-8 -*-
<%
import numpy as np
import networkx as nx
from sympy import Symbol

if 'experiment' in context.keys():
    integration = experiment.integration
    model = experiment.dynamics
    from tvbo.templates.base.utils import experiment_coupling
    coupling = experiment_coupling(experiment)
elif 'integration' in context.keys():
    integration = context['integration']
    model = context['model']
    coupling = context.get('coupling', None)
else:
    raise ValueError("No integration metadata found")

stochastic = integration.noise is not None
is_delayed = experiment.horizon > 1 if 'experiment' in context.keys() else integration.delayed
jaxcode_obj = lambda obj: model.render_equation(obj, format='jax')

# Identify output derived vars that depend on dfun arguments (must compute in integrate)
dfun_args = set(model.coupling_terms.keys()) | {'t', 'noise', 'stimulus'}
derived_var_names = list(model.derived_variables.keys())
svars = list(model.state_variables.keys())
output_vars = list(model.output) if model.output else svars
G = model.get_dependency_tree()

output_derived_in_integrate = []
for var_name in output_vars:
    if var_name in derived_var_names:
        var_sym = Symbol(var_name)
        if var_sym in G.nodes():
            if any(str(a) in dfun_args for a in nx.ancestors(G, var_sym)):
                output_derived_in_integrate.append(var_name)

# Collect all derived vars needed (dependencies) in topological order
all_integrate_derived_vars = []
for var_name in output_derived_in_integrate:
    var_sym = Symbol(var_name)
    if var_sym in G.nodes():
        for anc in nx.ancestors(G, var_sym):
            anc_name = str(anc)
            if anc_name in derived_var_names and anc_name not in all_integrate_derived_vars:
                all_integrate_derived_vars.append(anc_name)
        if var_name not in all_integrate_derived_vars:
            all_integrate_derived_vars.append(var_name)
# Sort topologically
if all_integrate_derived_vars:
    subG = G.subgraph([Symbol(n) for n in all_integrate_derived_vars if Symbol(n) in G.nodes()])
    all_integrate_derived_vars = [str(s) for s in nx.topological_sort(subG) if str(s) in all_integrate_derived_vars]

# Boundaries: a state variable is clamped only when its ``domain`` opts in via
# enforce == 'clamp' (the legacy ``boundaries`` slot is folded into ``domain``
# with enforce: clamp by the Dynamics loader). enforce 'none' (default) leaves
# the trajectory unconstrained, so a domain is pure metadata unless enforced.
from tvbo.utils import domain_enforcement
def _clamp_enforced(sv):
    dom = getattr(sv, "domain", None)
    return dom is not None and domain_enforcement(dom) == "clamp"
boundaries = [(sv.domain.lo, sv.domain.hi) for sv in model.state_variables.values() if _clamp_enforced(sv)]
get_lower_bound = lambda sv: float(sv.domain.lo) if _clamp_enforced(sv) and sv.domain.lo is not None else float('-inf')
get_upper_bound = lambda sv: float(sv.domain.hi) if _clamp_enforced(sv) and sv.domain.hi is not None else float('inf')
%>

## Helper that converts a array into a string that can be read as array again
<%def name = "array_input(array)" filter="trim">
        <%
        import numpy as np
        %>
        ## ${np.set_printoptions(threshold=sys.maxsize)}
        jnp.array(${np.array2string(array, separator = ",")})
</%def>

## TODO: Implement Stimulus

def integrate(state, weights, dt, params_integrate, delay_indices, external_input):
    """
    ${integration.method} Integration
    ${'=' * len(integration.method) + '='*len(" Integration")}
    """
<%
    from tvbo.templates.base.utils import referenced
    _scheme = "\n".join(
        [str(integration.update_expression.equation.rhs)]
        + [str(s.equation.rhs) for s in integration.intermediate_expressions.values()]
    )
%>\
    %if stochastic:
    t, noise = external_input
    % else:
    t, _ = external_input
    % if referenced(['noise'], _scheme):
    noise = 0
    % endif
    % endif

    ## dt = ${integration.step_size}

    params_dfun, params_cfun, params_stimulus = params_integrate

    history, current_state = state
    % if has_stimulus:
    stimulus = get_stimulus(t, params_stimulus)
    % elif referenced(['stimulus'], _scheme):
    stimulus = 0
    % endif

    ## Clip state if bounds are defined
    % if boundaries:
    inf = jnp.inf
    ## Clip by boundaries if present else set bounderies to +-inf -> no clip
    min_bounds = jnp.array(${[[[get_lower_bound(sv)]] for sv in model.state_variables.values()]})
    max_bounds = jnp.array(${[[[get_upper_bound(sv)]] for sv in model.state_variables.values()]})
    % endif

    cX = jax.vmap(cfun, in_axes=(None, -1, -1, None, None, None), out_axes=-1)(weights, history, current_state, params_cfun, delay_indices, t)

    dX0 = dfun(current_state, t, cX, params_dfun)
% if referenced(['X'], _scheme):

    X = current_state
% endif

    ## % for i, exp in enumerate(integration.intermediate_expressions.values()):
    ## inter_k${i+1} = ${exp.equation.rhs}
    ## ${exp.name} = dfun(inter_k${i+1}, cX, params_dfun)

    ## % endfor
    ## ## Calculate the state change dX
    ## dX = ${integration.update_expression.equation.rhs}

    ###
    % for k, step in integration.intermediate_expressions.items():
    # Calculate intermediate step ${k}
    ${k} = ${step.equation.rhs}
    ## self.integration_bound_and_clamp(${k})
    % if boundaries:
    ${k} = jnp.clip(${k}, min_bounds, max_bounds)
    % endif

    # Calculate derivative ${k}
    d${k} = dfun(${k}, t, cX, params_dfun)
    % endfor
    # Calculate the state change dX
    dX = ${integration.update_expression.equation.rhs}
    ## # Calculate the next state of X, including noise
    ## X_next = X + dX + noise + stimulus * dt
    ## self.integration_bound_and_clamp(X_next)
    ## return X_next
    ###
    next_state = current_state + (dX)${" + dt * stimulus" if has_stimulus else ''}${' + noise' if stochastic else ''}
    % if boundaries:
    next_state = jnp.clip(next_state, min_bounds, max_bounds)
    % endif

% if output_derived_in_integrate:
    # Compute output derived variables that depend on dfun arguments
    ## Unpack parameters needed for derived var computation
    _p = params_dfun
    ${', '.join(p.name for p in model.parameters.values())} = ${', '.join('_p.' + p.name for p in model.parameters.values())}
    ## First unpack state variables from next_state for derived var computation
    % for i, svar in enumerate(model.state_variables.keys()):
    ${svar} = next_state[${i}]
    % endfor
    ## Unpack coupling terms
    % for i, cterm in enumerate(model.coupling_terms.keys()):
    ${cterm} = cX[${i}]
    % endfor
    ## Emit model function definitions needed by derived variables
    % for f in (model.functions or {}).values():
    <%
        from tvbo.templates.base.utils import get_func_args
    %>
    def ${f.name}(${', '.join(get_func_args(f))}):
        return ${jaxcode_obj(f)}
    % endfor

    ## Compute derived variables in dependency order
    % for dv_name in all_integrate_derived_vars:
    <%
        dv = model.derived_variables[dv_name]
        rendered_eq = jaxcode_obj(dv)
    %>
    ${dv_name} = ${rendered_eq}
    % endfor

    ## Stack output derived vars into array
    _output_derived = jnp.stack([${', '.join(output_derived_in_integrate)}], axis=0)
% endif

## Return for scan: carry, result
% if is_delayed:
    <%
    from tvbo.templates.base.utils import gathered_state_indices
    cvar_list = gathered_state_indices(model, coupling)
    %>
    %if len(cvar_list) > 0:
    cvar = ${array_input(np.array(cvar_list))}
    % if small_dt:
    history = history.at[:, t, :].set(next_state[cvar, :])
    % else:
    _h = jnp.roll(history, -1, axis=1)
    history = _h.at[:, -1, :].set(next_state[cvar, :])
    % endif
    %else:
    ## When no coupling variables are defined, store all state variables in history
    % if small_dt:
    history = history.at[:, t, :].set(next_state)
    % else:
    _h = jnp.roll(history, -1, axis=1)
    history = _h.at[:, -1, :].set(next_state)
    % endif
    %endif
% endif
## Return for scan: (carry, output) - output may include derived vars
% if output_derived_in_integrate:
    % if monitor_node_coupling:
    return (history, next_state), (next_state, cX, _output_derived)
    % else:
    return (history, next_state), (next_state, _output_derived)
    % endif
% else:
    % if monitor_node_coupling:
    return (history, next_state), (next_state, cX)
    % else:
    return (history, next_state), next_state
    % endif
% endif


