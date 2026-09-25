<%!
    import numpy as np
    from tvbo.codegen import render_expression
    from tvbo.templates.base.utils import referenced

    # Generic pycode - pass parameters on each call. Use the 'numpy' format: TVB coupling
    # pre/post operate on ARRAYS (history states), so functions must be numpy (np.sin), not
    # scalar `math.sin` — the 'python' format emitted math.* which NameError'd / failed on arrays.
    pycode = lambda expr, parameters=None: render_expression(expr, format='numpy', parameters=parameters)
%>
<%
coupling = context['coupling']

_has_coupling = coupling is not None

if _has_coupling:
    # Collect coupling parameter names for use in expressions
    coupling_param_names = [par.name for par in coupling.parameters.values()] if coupling.parameters else []

    if coupling.sparse:
        base_class = 'SparseCoupling'
    else:
        base_class = 'Coupling'

    pre_expr = pycode(coupling.pre_expression.rhs, parameters=coupling_param_names)
    if '[0]' in pre_expr:
        pre_expr = pre_expr.replace('[', '[:, ')
        return_new_axis = "[:, np.newaxis]"
    else:
        return_new_axis = ""

    post_expr = pycode(coupling.post_expression.rhs, parameters=['gx'] + coupling_param_names)
%>
% if _has_coupling:
##
class ${coupling.name}(${base_class}):
    """
    This is a custom Coupling class generated from a template.
    It allows for specific pre and post expression definitions.
    """

    % for k, param in coupling.parameters.items():
    ${k} = NArray(
        label="${k}",
        default=np.array([${getattr(param, 'value', '0')},]),
        domain=Range(lo=${getattr(param.domain, 'lo', '0.0')},
                     hi=${getattr(param.domain, 'hi', '1.0')},
                     step=${getattr(param.domain, 'step', '0.01')}),
        doc="${getattr(param, 'description', '')}"
    )
    % endfor

    parameter_names = ${list(coupling.parameters)}
    pre_expr = "${pre_expr}"
    post_expr = "${post_expr}"

    def pre(self, x_i, x_j):
        """
        Pre-expression method.
        """
        % for param in referenced(coupling.parameters, pre_expr):
        ${param} = self.${param}
        % endfor
<%
    _loc = getattr(coupling, 'local_states', None) or []
    _loc = [_loc] if isinstance(_loc, str) else list(_loc)
    _inc = getattr(coupling, 'incoming_states', None) or []
    _inc = [_inc] if isinstance(_inc, str) else list(_inc)
%>\
        ## Bind state-variable symbols (e.g. theta_i/theta_j) to TVB's generic x_i (local) / x_j (incoming)
        % for st in referenced([f'{s}_i' for s in _loc], pre_expr):
        ${st} = x_i
        % endfor
        % for st in referenced([f'{s}_j' for s in _inc], pre_expr):
        ${st} = x_j
        % endfor

        pre = ${pre_expr}
        return pre${return_new_axis}

    def post(self, gx):
        """
        Post-expression method.
        """
        % for param in referenced(coupling.parameters, post_expr):
        ${param} = self.${param}
        % endfor

        post = ${post_expr}
        return post

    def __str__(self):
        return simple_gen_astr(self, "${" ".join(list(p.name for p in coupling.parameters.values()))}")
% endif
