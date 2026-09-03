# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Coupling (cfun) Template
==================================

Generates coupling classes for tvboptim.experimental.network_dynamics.

Context Variables (two modes):
- experiment: SimulationExperiment instance (experiment mode — renders all network couplings)
- coupling: single Coupling instance (standalone mode — renders one coupling class)

Output:
- Python class(es) inheriting from InstantaneousCoupling or DelayedCoupling
</%doc>
<%
from tvbo.codegen import render_expression
from tvbo.templates.base.utils import referenced
from tvbo.templates.tvboptim.utils import get_param_info, normalize_coupling_aliases, get_mode_layout, resolve_coupling_spec, parse_list_elements

# Two modes: experiment (full pipeline) or standalone (single coupling)
if 'experiment' in context.keys():
    network = experiment.network
    model = experiment.dynamics

    # Build coupling_inputs lookup: key -> {dimension, keys}
    coupling_inputs_info = {}
    if hasattr(model, 'coupling_inputs') and model.coupling_inputs:
        for ci_key, ci in model.coupling_inputs.items():
            dim = getattr(ci, 'dimension', 1) or 1
            keys = getattr(ci, 'keys', None)
            coupling_inputs_info[ci_key] = {'dimension': dim, 'keys': list(keys) if keys else None}

    # Resolved by TvboptimAdapter.resolve_couplings — see tvbo/adapters/tvboptim.py.
    from tvbo.adapters.tvboptim import TvboptimAdapter

    all_couplings = context.get('all_couplings') or TvboptimAdapter(experiment).resolve_couplings()

elif 'coupling' in context.keys():
    _standalone_coupling = context['coupling']
    model = None
    all_couplings = normalize_coupling_aliases({_standalone_coupling.name: _standalone_coupling}, model)
    coupling_inputs_info = {}

else:
    assert False, "cfun template requires 'experiment' or 'coupling' in context"

# Mode folding (number_of_modes>1): each coupling-variable carries n_modes per-node
# slots, folded into the state axis. A coupling that reads such a cvar emits one
# output per mode (each mode coupled independently across nodes via the existing
# 2-D matmul), so n_output == n_modes and the cvar resolves to its mode slots.
n_modes, _mode_slot_names, _var_slots = get_mode_layout(model) if model is not None else (1, [], {})

# Build func_name -> ci_name mapping.
# Resolution order:
#   1. Explicit source on CouplingInput (ci.source == func_name)
#   2. Same name (ci_name == func_name)
#   3. Single CI + single func → auto-map
#   4. Same count → positional zip
_func_to_ci = {}
if coupling_inputs_info and all_couplings:
    _funcs = list(all_couplings.keys())
    _ci_names = list(coupling_inputs_info.keys())

    # 1. Explicit source attribute
    if hasattr(model, 'coupling_inputs') and model is not None and model.coupling_inputs:
        for ci_key, ci in model.coupling_inputs.items():
            src = getattr(ci, 'source', None)
            if src and src in all_couplings:
                _func_to_ci[src] = ci_key

    # 2. Same name match
    for fn in _funcs:
        if fn not in _func_to_ci and fn in coupling_inputs_info:
            _func_to_ci[fn] = fn

    # 3/4. Fallback for remaining unmapped functions
    _unmapped_funcs = [f for f in _funcs if f not in _func_to_ci]
    _unmapped_cis = [c for c in _ci_names if c not in _func_to_ci.values()]
    if len(_unmapped_funcs) == 1 and len(_unmapped_cis) == 1:
        _func_to_ci[_unmapped_funcs[0]] = _unmapped_cis[0]
    elif len(_unmapped_funcs) == len(_unmapped_cis):
        for _ci, _fn in zip(_unmapped_cis, _unmapped_funcs):
            _func_to_ci.setdefault(_fn, _ci)
%>
% for coupling_key, coupling in all_couplings.items():
<%
    # Coupling resolution lives in the tvboptim Python layer (resolve_coupling_spec);
    # this template only emits. Unpack the spec to the local names the emission below
    # already uses. Expression rendering (jaxcode) stays here — it needs the printer.
    spec = resolve_coupling_spec(coupling, coupling_key, model, coupling_inputs_info, _func_to_ci, n_modes)
    n_output = spec['n_output']
    has_delay = spec['has_delay']
    param_names, param_defaults, param_shapes = spec['param_names'], spec['param_defaults'], spec['param_shapes']
    incoming_states = spec['incoming_states']
    local_states = spec['local_states']
    pre_expr = spec['pre_expr']
    post_expr = spec['post_expr']
    mode_coupling = spec['mode_coupling']
    pre_is_list = spec['pre_is_list']
    pre_terms = spec['pre_terms']
    n_pre = spec['n_pre']
    vectorized = spec['vectorized']
    vec_states = spec['vec_states']
    vec_identity = spec['vec_identity']
    edge_params = spec['edge_params']
    class_name = spec['class_name']
    base_class = spec['base_class']
    _interp_kw = spec['interp_kw']
    _post_aliases_i = spec['post_aliases_i']
    post_is_list = spec['post_is_list']
    _state_aliases_j = spec['state_aliases_j']
    _state_aliases_i = spec['state_aliases_i']
    _gx_symbols = spec['gx_symbols']
    all_symbols = spec['all_symbols']
    description = spec['description']
    jaxcode = lambda expr: render_expression(expr, format='jax', parameters=all_symbols)
    _uses_local = spec['uses_local']    # pre() reads a target-local state (resolved in resolve_coupling_spec)
    # N_OUTPUT_STATES tracks pre()'s leading axis, so a factored pre that post() recombines (n_pre > 1, n_output == 1) reduces over n_pre.
    n_output_states = n_pre if (n_pre > 1 and n_output == 1) else n_output
%>

class ${class_name}(${base_class}):
    """${class_name} coupling function."""

    N_OUTPUT_STATES = ${n_output_states}
    % if _uses_local and not vectorized:
    PRE_USES_LOCAL = True    # pre() reads target-local state (x_i/<state>_i); base class aligns local_states to the message axes
    % endif
    % if edge_params:
    EDGE_PARAMS = (${', '.join(repr(p) for p in edge_params)},)    # graph-shaped (n_nodes, n_nodes) params; base class aligns them to the message axes
    % endif

    DEFAULT_PARAMS = Bunch(
        % for name in param_names:
        ${name}=${param_defaults.get(name, 1.0)},
        % endfor
        % if not param_names:
        G=1.0,
        % endif
    )

    def __init__(self, **kwargs):
        % if vectorized:
## tvboptim keeps incoming (source) and local (target) states distinct and post() reads only local, so a target-local alias must register both.
        super().__init__(${_interp_kw}incoming_states=${vec_states}${''.join([', local_states=' + str(vec_states)] if _post_aliases_i else [])}, **kwargs)
        % elif incoming_states:
        super().__init__(${_interp_kw}incoming_states=${incoming_states}${''.join([', local_states=' + str(local_states)] if local_states else [])}, **kwargs)
        % elif local_states:
        super().__init__(${_interp_kw}local_states=${local_states}, **kwargs)
        % else:
        super().__init__(${_interp_kw}**kwargs)
        % endif

    % if vectorized and vec_identity:
## Incoming-only path: return the source messages; the base class reduces `pre @ weights`.
    def pre(self, incoming_states, local_states, params):
        return incoming_states
    % elif vectorized:
## Source-only pre: stack f(stateⱼ) per node into [n_pre, n_source]; base class W-reduces.
    def pre(self, incoming_states, local_states, params):
<% _pre_ref = ' '.join(str(t) for t in pre_terms) %>\
        % for name in referenced(param_names, _pre_ref):
        ${name} = params.${name}
        % endfor
        % for k, s in enumerate(vec_states):
        % if referenced([s], _pre_ref):
        ${s} = incoming_states[${k}]
        % endif
        % if referenced([s + '_j'], _pre_ref):
        ${s}_j = incoming_states[${k}]
        % endif
        % endfor
        return jnp.stack([${', '.join(jaxcode(t) for t in pre_terms)}], axis=0)
    % elif pre_expr:
    def pre(self, incoming_states, local_states, params):
<% _pre_ref = str(pre_expr.rhs) %>\
        % for name in referenced(param_names, _pre_ref):
        ${name} = params.${name}
        % endfor
## Assign incoming state variables (skip when name collides with local)
        % for i, state_name in enumerate(incoming_states):
        % if state_name not in local_states and referenced([state_name], _pre_ref):
        ${state_name} = incoming_states[${i}]
        % endif
        % endfor
## The base class aligns local states to the message axes, so pre() indexes straight in and stays elementwise.
        % for i, state_name in enumerate(local_states):
        % if state_name not in incoming_states and referenced([state_name], _pre_ref):
        ${state_name} = local_states[${i}]
        % endif
        % endfor
<%
        # Alias resolution for coupling expressions.
        # Supports three naming conventions:
        #   1. x_i / x_j          — generic placeholders (database coupling functions)
        #   2. theta_i / theta_j   — state-subscript notation (mathematical)
        #   3. incoming_states / local_states — literal parameter names
        _pre_rhs = str(pre_expr.rhs) if pre_expr else ''
        _need_xj = 'x_j' in _pre_rhs and 'x_j' not in incoming_states
        _need_xi = 'x_i' in _pre_rhs and 'x_i' not in local_states
        _need_incoming = 'incoming_states' in _pre_rhs
        _need_local = 'local_states' in _pre_rhs
%>
## State-subscript aliases: e.g. theta_j = incoming_states[0], theta_i = local_states[0]
        % for alias_name, idx in _state_aliases_j:
        ${alias_name} = incoming_states[${idx}]
        % endfor
        % for alias_name, idx in _state_aliases_i:
        ${alias_name} = local_states[${idx}]
        % endfor
        % if _need_xj and incoming_states:
        % if mode_coupling:
        ## Mode fold: keep all n_modes gathered slots (leading axis) so each mode
        ## is reduced with the connectome independently → per-mode coupling output.
        x_j = incoming_states
        % else:
        x_j = incoming_states[0]
        % endif
        % endif
        % if _need_xi and local_states:
        x_i = local_states[0]
        % endif
        % if _need_local and local_states:
        local_states = local_states[0]
        % endif
        % if _need_incoming and incoming_states:
        incoming_states = incoming_states[0]
        % endif
<%
        # A list pre_expression declares n_pre separate reductions (e.g. the
        # angle-addition decomposition [sin(θⱼ), cos(θⱼ)]). Each term is per-edge
        # [N_target, N_source]; stacking yields 3D [n_pre, N_target, N_source] so
        # the base class W-reduces every term in one weighted sum. n_pre is the
        # reduction count, independent of n_output (the model's input dimension).
        if pre_is_list:
            rendered = [jaxcode(e) for e in pre_terms]
            pre_code = 'jnp.stack([' + ', '.join(rendered) + '], axis=0)'
        else:
            pre_code = jaxcode(pre_expr.rhs)
%>
        coupling_term = ${pre_code}
        % if pre_is_list:
## Stacked list pre: already 3D [n_pre, N_target, N_source].
        return coupling_term
        % elif (incoming_states and local_states) or has_delay:
## Prepend the n_output axis whatever the message's rank, so the base class reduces every output component over the edges.
        return jnp.expand_dims(coupling_term, axis=0)
        % elif mode_coupling:
## Mode-fold: coupling_term already carries the leading per-mode axis [n_modes, n_source].
        return coupling_term
        % else:
## Incoming-only node message: add the leading n_output axis → [n_output, n_source].
        return jnp.expand_dims(coupling_term, axis=0)
        % endif
    % endif

    def post(self, summed_inputs, local_states, params):
<% _post_ref = str(post_expr.rhs) if post_expr else 'G * gx' %>\
        % for name in referenced(param_names, _post_ref):
        ${name} = params.${name}
        % endfor
        % if 'G' not in param_names and referenced(['G'], _post_ref):
        G = params.G if hasattr(params, 'G') else 1.0
        % endif
        % if referenced(['gx'], _post_ref):
        gx = summed_inputs
        % endif
## Recombination case: n_pre source reductions collapse into a SINGLE coupling
## output (n_output == 1), e.g. the Kuramoto angle-addition identity. Expose the
## named components gx_0, gx_1, … so post_expression can recombine them. When
## n_output > 1 the list pre is a multi-output coupling (each reduction is its
## own output, e.g. [S_e*wLRE, S_e*wFFI] → c_lre, c_ffi); there gx stays the full
## [n_output, n_nodes] stack and post passes it through unchanged.
        % if n_pre > 1 and n_output == 1:
        % for k in range(n_pre):
        gx_${k} = summed_inputs[${k}]
        % endfor
        % endif
## Target (post-synaptic) state aliases referenced in post_expression (e.g. θᵢ).
## Bound whenever present — not only in the recombination case — so any
## post_expression that uses a `{state}_i` alias resolves it.
        % for alias_name, idx in _post_aliases_i:
        ${alias_name} = local_states[${idx}]
        % endfor
        % if post_expr and post_is_list:
        return jnp.stack([${', '.join(jaxcode(e) for e in parse_list_elements(str(post_expr.rhs).strip()))}], axis=0)
        % elif post_expr and n_pre > 1 and n_output == 1:
## Scalar recombination of the n_pre reductions → per-node [n_nodes];
## add leading axis to return [n_output=1, n_nodes].
        return (${jaxcode(post_expr.rhs)})[jnp.newaxis, :]
        % elif post_expr:
        return ${jaxcode(post_expr.rhs)}
        % else:
        return G * gx
        % endif

% endfor
