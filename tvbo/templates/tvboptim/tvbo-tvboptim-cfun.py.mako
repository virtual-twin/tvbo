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
from tvbo.export.code import render_expression
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

    # JAX code helper
    all_symbols = param_names + incoming_states + local_states + ['gx', 'G']
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
        % for i, state_name in enumerate(incoming_states):
        ${state_name} = incoming_states[${i}]
        % endfor
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
        % if has_delay:
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
