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
from tvbo.templates.tvboptim.utils import get_param_info, normalize_coupling_aliases

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

    # Get all couplings from network.coupling, fall back to experiment-level coupling
    all_couplings = {}
    if hasattr(network, 'coupling') and network.coupling:
        if hasattr(network.coupling, 'items'):
            all_couplings = dict(network.coupling.items())
        elif hasattr(network.coupling, 'keys'):
            all_couplings = {k: network.coupling[k] for k in network.coupling.keys()}
    if not all_couplings and getattr(experiment, 'coupling', None):
        _exp_c = experiment.coupling
        if hasattr(_exp_c, 'items'):
            all_couplings = dict(_exp_c.items())
        else:
            all_couplings = {_exp_c.name or 'coupling': _exp_c}
    all_couplings = normalize_coupling_aliases(all_couplings, model)

elif 'coupling' in context.keys():
    _standalone_coupling = context['coupling']
    model = None
    all_couplings = normalize_coupling_aliases({_standalone_coupling.name: _standalone_coupling}, model)
    coupling_inputs_info = {}

else:
    assert False, "cfun template requires 'experiment' or 'coupling' in context"

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

def parse_list_elements(rhs_str):
    """Parse a list literal string into elements, respecting nesting."""
    inner = rhs_str[1:-1]  # Remove [ and ]
    elements = []
    depth = 0
    current = []
    for c in inner:
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == ',' and depth == 0:
            elements.append(''.join(current).strip())
            current = []
            continue
        current.append(c)
    if current:
        elements.append(''.join(current).strip())
    return elements
%>
% for coupling_key, coupling in all_couplings.items():
<%
    # Get dimension from coupling_inputs — translate function name to ci name
    _ci_key = _func_to_ci.get(coupling_key, coupling_key)
    ci_info = coupling_inputs_info.get(_ci_key, {'dimension': 1, 'keys': None})
    n_output = ci_info['dimension']

    # Coupling metadata
    has_delay = getattr(coupling, 'delayed', False)

    # Extract parameter info using shared utility
    param_names, param_defaults, param_shapes = get_param_info(coupling.parameters if hasattr(coupling, 'parameters') else None)

    incoming_states = getattr(coupling, 'incoming_states', None) or []
    if isinstance(incoming_states, str):
        incoming_states = [incoming_states]
    incoming_states = list(incoming_states) if incoming_states else []

    local_states = getattr(coupling, 'local_states', None) or []
    if isinstance(local_states, str):
        local_states = [local_states]
    local_states = list(local_states) if local_states else []

    pre_expr = coupling.pre_expression if hasattr(coupling, 'pre_expression') and coupling.pre_expression else None
    post_expr = coupling.post_expression if hasattr(coupling, 'post_expression') and coupling.post_expression else None

    # Infer incoming_states from pre_expression if not explicitly given
    # Match state variable names referenced in the expression against model svars
    if not incoming_states and not local_states and pre_expr:
        svar_names = set()
        if model and hasattr(model, 'state_variables') and model.state_variables:
            svar_names = {sv if isinstance(sv, str) else getattr(sv, 'name', str(sv))
                          for sv in (model.state_variables.keys()
                                     if hasattr(model.state_variables, 'keys')
                                     else model.state_variables)}
        pre_rhs = str(pre_expr.rhs) if pre_expr else ''
        for sv in svar_names:
            if sv in pre_rhs:
                incoming_states.append(sv)
        if not incoming_states:
            # pre_expression uses generic placeholders (e.g. x_j) —
            # resolve to the coupling_variable state(s) from the model
            cvars = []
            if model and hasattr(model, 'state_variables') and model.state_variables:
                for sv_name, sv_obj in model.state_variables.items():
                    if getattr(sv_obj, 'coupling_variable', False):
                        cvars.append(sv_name)
            incoming_states = cvars if cvars else [pre_rhs.strip()]

    # Parse pre_expression into one or more terms. A list literal
    # `[f(source), g(source), ...]` declares several source-only reductions —
    # the angle-addition decomposition of a phase coupling — each reduced by W
    # and recombined in post(). `n_pre` is the number of W-reductions, distinct
    # from `n_output` (the coupling-input dimension the model consumes).
    _pre_rhs0 = str(pre_expr.rhs).strip() if pre_expr else ''
    pre_is_list = _pre_rhs0.startswith('[') and _pre_rhs0.endswith(']')
    pre_terms = parse_list_elements(_pre_rhs0) if pre_is_list else ([_pre_rhs0] if _pre_rhs0 else [])
    n_pre = len(pre_terms)

    # A pre_expression is "source-only" when it references source states
    # (incoming, or `{state}_j` aliases) but no target states (local, or
    # `{state}_i` aliases). Such couplings reduce to a per-node pre() value plus
    # a single matmul with W, so instantaneous ones take the vectorized fast
    # path; delayed ones keep the per-edge form (delays are inherently per-edge)
    # but stay numerically identical.
    def _refs_target(_rhs):
        if 'x_i' in _rhs or 'local_states' in _rhs:
            return True
        for s in local_states:
            if f'{s}_i' in _rhs:
                return True
            if s not in incoming_states and s in _rhs:
                return True
        return False
    pre_source_only = bool(pre_expr) and not _refs_target(_pre_rhs0)

    # Vectorized mode: pre() returns [n_pre, n_nodes] (per-node) so the base
    # class reduces with a single matmul `pre @ weights` instead of forming the
    # dense [.., N, N] per-edge tensor. This matmul is only equivalent to the
    # per-edge reduction `Σ_k pre·W[:, k]` when the connectome is SYMMETRIC
    # (the two differ by W vs Wᵀ for directed weights), so it is NOT enabled
    # automatically for source-only phase couplings. Instead we enable it for:
    #   (a) the legacy identity case (local states, no incoming) — unchanged
    #       prior behaviour, and
    #   (b) an explicit `vectorized: true` opt-in on the coupling (caller asserts
    #       a symmetric connectome).
    # The angle-addition decomposition (source-only pre) otherwise takes the
    # per-edge path below: it stays exact for ANY connectome while still
    # eliminating the expensive per-edge transcendental evaluations.
    vectorized = getattr(coupling, 'vectorized', False)
    if not vectorized and local_states and not incoming_states:
        vectorized = True
    vec_states = list(dict.fromkeys(incoming_states + local_states))

    # Class name = coupling key (cleaned for Python identifier)
    class_name = coupling_key.replace(' ', '').replace('-', '')
    base_class = 'DelayedCoupling' if has_delay else 'InstantaneousCoupling'
    # Differentiable (interpolated) delays: OPT-IN only. The kwarg is emitted
    # solely when the coupling explicitly sets ``interpolate_delays: true`` — it
    # requires the differentiable-delays tvboptim API. Normal delayed coupling
    # (the vast majority) emits no kwarg and uses the stock DelayedCoupling, so
    # it keeps working against released tvboptim. InstantaneousCoupling never
    # takes the kwarg.
    _interp_kw = 'interpolate_delays=True, ' if (has_delay and bool(getattr(coupling, "interpolate_delays", False))) else ''

    # Target-state aliases referenced in post_expression (e.g. theta_i), bound
    # in post() from the per-node local states so the recombination step can
    # use the post-synaptic state.
    _post_aliases_i = []
    post_is_list = False
    if post_expr:
        _post_rhs_str = str(post_expr.rhs).strip()
        post_is_list = _post_rhs_str.startswith('[') and _post_rhs_str.endswith(']')
        _post_state_list = vec_states if vectorized else local_states
        for idx, s in enumerate(_post_state_list):
            si = f'{s}_i'
            if si in _post_rhs_str:
                _post_aliases_i.append((si, idx))

    # Build state-subscript aliases for mathematical notation in expressions.
    # Enables e.g. pre_expression: sin(theta_j - theta_i) where:
    #   {state}_j -> incoming_states[idx]  (source / pre-synaptic state)
    #   {state}_i -> local_states[idx]     (target / post-synaptic state, reshaped)
    _state_aliases_j = []  # (alias_name, index)
    _state_aliases_i = []
    if pre_expr:
        _pre_rhs_str = str(pre_expr.rhs)
        for idx, s in enumerate(incoming_states):
            sj = f'{s}_j'
            if sj in _pre_rhs_str:
                _state_aliases_j.append((sj, idx))
        for idx, s in enumerate(local_states):
            si = f'{s}_i'
            if si in _pre_rhs_str:
                _state_aliases_i.append((si, idx))
    _alias_symbols = [a[0] for a in _state_aliases_j] + [a[0] for a in _state_aliases_i]
    # Named reduction components: post_expression refers to gx_0, gx_1, … (one
    # per pre term) plus the target-state aliases bound in post().
    _gx_symbols = ['gx_%d' % k for k in range(n_pre)] if n_pre > 1 else []
    _post_alias_symbols = [a[0] for a in _post_aliases_i]

    # JAX code helper
    all_symbols = param_names + incoming_states + local_states + ['gx', 'G', 'x_i', 'x_j', 'incoming_states', 'local_states'] + _alias_symbols + _gx_symbols + _post_alias_symbols
    jaxcode = lambda expr: render_expression(expr, format='jax', parameters=all_symbols)

    # Description
    description = coupling.description if hasattr(coupling, 'description') and coupling.description else 'Auto-generated coupling function.'
%>

class ${class_name}(${base_class}):
    """${class_name} coupling function."""

    N_OUTPUT_STATES = ${n_output}

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
        super().__init__(${_interp_kw}local_states=${vec_states}, **kwargs)
        % elif incoming_states:
        super().__init__(${_interp_kw}incoming_states=${incoming_states}${''.join([', local_states=' + str(local_states)] if local_states else [])}, **kwargs)
        % elif local_states:
        super().__init__(${_interp_kw}local_states=${local_states}, **kwargs)
        % else:
        super().__init__(${_interp_kw}**kwargs)
        % endif

    % if vectorized and not pre_expr:
    def pre(self, incoming_states, local_states, params):
        return local_states
    % elif vectorized and pre_expr:
## Source-only vectorized pre(): evaluate each pre term on the PER-NODE state
## (local_states holds the union of source+target states, [n_states, n_nodes])
## and stack to [n_pre, n_nodes] so the base class reduces with a single matmul
## `pre @ weights`. The W-sum over sources is what turns the per-node sin/cos
## values into Σⱼ wᵢⱼ·f(stateⱼ).
    def pre(self, incoming_states, local_states, params):
        % for name in param_names:
        ${name} = params.${name}
        % endfor
        % for k, s in enumerate(vec_states):
        ${s} = local_states[${k}]
        ${s}_j = local_states[${k}]
        % endfor
        return jnp.stack([${', '.join(jaxcode(t) for t in pre_terms)}], axis=0)
    % elif pre_expr:
    def pre(self, incoming_states, local_states, params):
        % for name in param_names:
        ${name} = params.${name}
        % endfor
## Assign incoming state variables (skip when name collides with local)
        % for i, state_name in enumerate(incoming_states):
        % if state_name not in local_states:
        ${state_name} = incoming_states[${i}]
        % endif
        % endfor
## Assign local state variables (skip when name collides with incoming)
## incoming_states are per-edge: [N_target, N_source] (both Delayed and Instantaneous).
## local_states are per-node: [N_nodes].  Reshape to [N_nodes, 1] for correct
## broadcasting: result[j,k] = f(local_j, incoming_j_k).
        % for i, state_name in enumerate(local_states):
        % if state_name not in incoming_states:
        % if incoming_states:
        ${state_name} = local_states[${i}][:, jnp.newaxis]
        % else:
        ${state_name} = local_states[${i}]
        % endif
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
        % if incoming_states:
        ${alias_name} = local_states[${idx}][:, jnp.newaxis]
        % else:
        ${alias_name} = local_states[${idx}]
        % endif
        % endfor
        % if _need_xj and incoming_states:
        x_j = incoming_states[0]
        % endif
        % if _need_xi and local_states:
        % if incoming_states:
        x_i = local_states[0][:, jnp.newaxis]
        % else:
        x_i = local_states[0]
        % endif
        % endif
        % if _need_local and local_states:
        % if incoming_states:
        local_states = local_states[0][:, jnp.newaxis]
        % else:
        local_states = local_states[0]
        % endif
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
        % elif incoming_states and local_states:
## Per-edge output: ensure 3D [n_output, N_target, N_source] for weighted sum
        return coupling_term[jnp.newaxis, :, :]
        % elif has_delay:
        return coupling_term[jnp.newaxis, :, :]
        % else:
        return coupling_term
        % endif
    % endif

    def post(self, summed_inputs, local_states, params):
        % for name in param_names:
        ${name} = params.${name}
        % endfor
        % if 'G' not in param_names:
        G = params.G if hasattr(params, 'G') else 1.0
        % endif
        gx = summed_inputs
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
