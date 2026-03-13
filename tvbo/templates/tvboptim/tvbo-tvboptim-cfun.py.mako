# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Coupling (cfun) Template
==================================

Generates coupling classes for tvboptim.experimental.network_dynamics.

Context Variables:
- experiment: SimulationExperiment instance (required)

Each entry in network.coupling generates a coupling class.
The key in network.coupling becomes the key used in Network(coupling={key: instance}).
The key must also exist in dynamics.coupling_inputs for dimension/keys info.

Output:
- Python class(es) inheriting from InstantaneousCoupling or DelayedCoupling
</%doc>
<%
from tvbo.codegen import render_expression
from tvbo.templates.tvboptim.utils import get_param_info

# Get network and model from experiment
assert 'experiment' in context.keys(), "experiment required for cfun template"
network = experiment.network
model = experiment.dynamics

# Build coupling_inputs lookup: key -> {dimension, keys}
coupling_inputs_info = {}
if hasattr(model, 'coupling_inputs') and model.coupling_inputs:
    for ci_key, ci in model.coupling_inputs.items():
        dim = getattr(ci, 'dimension', 1) or 1
        keys = getattr(ci, 'keys', None)
        coupling_inputs_info[ci_key] = {'dimension': dim, 'keys': list(keys) if keys else None}

# Get all couplings from network.coupling
all_couplings = {}
if hasattr(network, 'coupling') and network.coupling:
    if hasattr(network.coupling, 'items'):
        all_couplings = dict(network.coupling.items())
    elif hasattr(network.coupling, 'keys'):
        all_couplings = {k: network.coupling[k] for k in network.coupling.keys()}

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
    # Get dimension from coupling_inputs (key must match)
    ci_info = coupling_inputs_info.get(coupling_key, {'dimension': 1, 'keys': None})
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
        if hasattr(model, 'state_variables') and model.state_variables:
            svar_names = {sv if isinstance(sv, str) else getattr(sv, 'name', str(sv))
                          for sv in (model.state_variables.keys()
                                     if hasattr(model.state_variables, 'keys')
                                     else model.state_variables)}
        pre_rhs = str(pre_expr.rhs) if pre_expr else ''
        for sv in svar_names:
            if sv in pre_rhs:
                incoming_states.append(sv)
        if not incoming_states:
            # Fallback: use the pre_expression rhs itself as an incoming state name
            incoming_states = [pre_rhs.strip()]

    # Vectorized mode: returns local_states from pre() for matmul optimization
    vectorized = getattr(coupling, 'vectorized', False)
    if not vectorized and local_states and not incoming_states:
        vectorized = True

    # Class name = coupling key (cleaned for Python identifier)
    class_name = coupling_key.replace(' ', '').replace('-', '')
    base_class = 'DelayedCoupling' if has_delay else 'InstantaneousCoupling'

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

    # JAX code helper
    all_symbols = param_names + incoming_states + local_states + ['gx', 'G', 'x_i', 'x_j', 'incoming_states', 'local_states'] + _alias_symbols
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
        super().__init__(local_states=${local_states}, **kwargs)
        % elif incoming_states:
        super().__init__(incoming_states=${incoming_states}${''.join([', local_states=' + str(local_states)] if local_states else [])}, **kwargs)
        % elif local_states:
        super().__init__(local_states=${local_states}, **kwargs)
        % else:
        super().__init__(**kwargs)
        % endif

    % if vectorized:
    def pre(self, incoming_states, local_states, params):
        return local_states
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
        pre_rhs = str(pre_expr.rhs).strip()
        is_list = pre_rhs.startswith('[') and pre_rhs.endswith(']')
        if is_list and n_output > 1:
            elements = parse_list_elements(pre_rhs)
            rendered = [jaxcode(e) for e in elements]
            pre_code = 'jnp.stack([' + ', '.join(rendered) + '], axis=0)'
        else:
            pre_code = jaxcode(pre_expr.rhs)
%>
        coupling_term = ${pre_code}
        % if incoming_states and local_states:
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
        % if post_expr:
        return ${jaxcode(post_expr.rhs)}
        % else:
        return G * gx
        % endif

% endfor
