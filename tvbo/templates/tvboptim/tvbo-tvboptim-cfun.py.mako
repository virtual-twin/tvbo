# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Coupling (cfun) Template
==================================

Generates coupling classes for tvboptim.experimental.network_dynamics.

Context Variables:
- experiment: SimulationExperiment instance (optional)
- coupling: Coupling instance (required if no experiment)
- model: Dynamics instance (for n_output info)

Output:
- Python class(es) inheriting from InstantaneousCoupling or DelayedCoupling
</%doc>
<%
from tvbo.export.code import render_expression

# Get coupling and model from context
if 'experiment' in context.keys():
    coupling = experiment.coupling
    model = experiment.local_dynamics
else:
    coupling = context['coupling']
    model = context.get('model', None)

# Determine if coupling is delayed
has_delay = hasattr(coupling, 'delayed') and coupling.delayed

# Get coupling parameters
coupling_params = list(coupling.parameters.values()) if hasattr(coupling, 'parameters') and coupling.parameters else []
param_names = [p.name for p in coupling_params]
param_defaults = {p.name: float(p.value) if p.value is not None else 1.0
                  for p in coupling_params}

# Get incoming/local states
incoming_states = getattr(coupling, 'incoming_states', None) or []
if isinstance(incoming_states, str):
    incoming_states = [incoming_states]
incoming_states = list(incoming_states) if incoming_states else []

local_states = getattr(coupling, 'local_states', None) or []
if isinstance(local_states, str):
    local_states = [local_states]
local_states = list(local_states) if local_states else []

# Get pre and post expressions
pre_expr = coupling.pre_expression if hasattr(coupling, 'pre_expression') and coupling.pre_expression else None
post_expr = coupling.post_expression if hasattr(coupling, 'post_expression') and coupling.post_expression else None

# Check for vectorized mode - returns local_states from pre() for matmul optimization
# Explicitly set via 'vectorized: true' or inferred when local_states specified without incoming_states
vectorized = getattr(coupling, 'vectorized', False)
if not vectorized and local_states and not incoming_states:
    # Infer vectorized mode when only local_states specified
    vectorized = True

# Class name and base class
class_name = coupling.name.replace(' ', '').replace('-', '') if hasattr(coupling, 'name') and coupling.name else 'GeneratedCoupling'
base_class = 'DelayedCoupling' if has_delay else 'InstantaneousCoupling'

# Number of output states - infer from:
# 1. coupling_inputs dimension in local_dynamics (preferred, declarative)
# 2. pre_expression list length (if it's a Python list literal like "[a, b]")
# 3. Default to 1
n_output = 1
if model and hasattr(model, 'coupling_inputs') and model.coupling_inputs:
    # Get max dimension from all coupling_inputs
    for ci_name, ci in model.coupling_inputs.items():
        dim = getattr(ci, 'dimension', 1)
        if dim and dim > n_output:
            n_output = dim
elif pre_expr and hasattr(pre_expr, 'rhs') and pre_expr.rhs:
    # Fallback: count list elements if pre_expression is a list literal "[a, b, c]"
    rhs = str(pre_expr.rhs).strip()
    if rhs.startswith('[') and rhs.endswith(']'):
        # Count commas at top level (not inside nested brackets)
        depth = 0
        count = 1
        for c in rhs[1:-1]:
            if c in '([{':
                depth += 1
            elif c in ')]}':
                depth -= 1
            elif c == ',' and depth == 0:
                count += 1
        n_output = count

# All symbol names: params + incoming_states + local_states + common names
# These must be passed to render_expression so they're parsed as Symbols, not as products
all_symbol_names = param_names + incoming_states + local_states + ['gx', 'G']

# JAX code generation helper - pass all symbols as parameters to prevent implicit multiplication
jaxcode = lambda expr: render_expression(expr, format='jax', parameters=all_symbol_names)
%>

class ${class_name}(${base_class}):
    """${class_name} coupling function.

    ${coupling.description if hasattr(coupling, 'description') and coupling.description else 'Auto-generated coupling function.'}
    """

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
        # Vectorized mode: only local_states needed for matmul optimization
        super().__init__(
            local_states=${local_states},
            **kwargs
        )
        % elif incoming_states:
        super().__init__(
            incoming_states=${incoming_states},
            % if local_states:
            local_states=${local_states},
            % endif
            **kwargs
        )
        % elif local_states:
        super().__init__(
            local_states=${local_states},
            **kwargs
        )
        % else:
        super().__init__(**kwargs)
        % endif

    % if vectorized:
    def pre(self, incoming_states, local_states, params):
        """Return local states to trigger vectorized mode.

        By returning [n_local, n_nodes] (2D), we trigger the vectorized
        path which uses matmul instead of per-edge ops.
        """
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
    # Check if pre_expression is a list literal (e.g., "[a, b, c]")
    # If so, convert to jnp.stack for proper multi-output coupling
    pre_rhs = str(pre_expr.rhs).strip()
    is_list_literal = pre_rhs.startswith('[') and pre_rhs.endswith(']')
    if is_list_literal and n_output > 1:
        # Parse list elements and generate jnp.stack
        inner = pre_rhs[1:-1]  # Remove [ and ]
        # Split by comma at top level (respecting nested parentheses)
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
        # Generate jnp.stack expression
        rendered_elements = [jaxcode(elem) for elem in elements]
        pre_code = 'jnp.stack([' + ', '.join(rendered_elements) + '], axis=0)'
    else:
        pre_code = jaxcode(pre_expr.rhs)
%>
        coupling_term = ${pre_code}
        % if has_delay:
        # Return as [1, n_nodes_target, n_nodes_source] for matrix multiplication
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
