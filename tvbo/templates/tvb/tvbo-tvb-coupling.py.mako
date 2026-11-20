<%!
    import numpy as np
    from tvbo.codegen import render_expression
    from tvbo.templates.base.utils import coupling_bindings, docstring_text, referenced

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
    # TVB hands pre() the gathered states (model.cvar, the coupling's incoming_states) on axis 1 of a dense history and axis 0 of a sparse one.
    _row = (lambda k: f'[{k}]') if coupling.sparse else (lambda k: f'[:, {k}]')
    _b = coupling_bindings(context['experiment'].dynamics if 'experiment' in context.keys() else context.get('model'), coupling)
    rows_j = [(s, _b['cvar_index'][s]) for s in _b['bare']] + _b['pre_j']
    rows_i = [(alias, _b['cvar_index'].get(alias[:-2])) for alias, _ in _b['pre_i']]
    for alias, k in rows_i:
        if k is None:
            raise NotImplementedError(
                f"coupling {coupling.name!r} reads the target's own {alias[:-2]!r} as `{alias}`, and the tvb backend hands a coupling only "
                f"the gathered states {_b['cvar_names']}, so it has no row for it. List {alias[:-2]!r} in the coupling's incoming_states, "
                "or run it on tvboptim or jax."
            )
    indexes_rows = '[0]' in pre_expr
    if indexes_rows and not coupling.sparse:
        pre_expr = pre_expr.replace('[', '[:, ')
    return_new_axis = ('[np.newaxis]' if coupling.sparse else '[:, np.newaxis]') if (indexes_rows or rows_j or rows_i) else ''

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
        doc="${docstring_text(getattr(param, 'description', ''))}"
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
        ## Each source state, bare or as <s>_j, reads its own gathered row of x_j, and each <s>_i the target's row of x_i.
        % for name, k in rows_j:
        ${name} = x_j${_row(k)}
        % endfor
        % for name, k in rows_i:
        ${name} = x_i${_row(k)}
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
