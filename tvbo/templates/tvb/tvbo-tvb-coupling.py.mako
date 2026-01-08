<%!
    import numpy as np
    from tvbo.export.code import render_expression
    from tvbo.knowledge.simulation.equations import _clash1

    # Generic pycode - pass parameters on each call
    pycode = lambda expr, parameters=None: render_expression(expr, format='python', parameters=parameters)
%>
<%
if 'experiment' in context.keys():
    coupling = context['experiment'].coupling.metadata
else:
    coupling = context['coupling'].metadata

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
        % for param in coupling.parameters:
        ${param} = self.${param}
        % endfor

        pre = ${pre_expr}
        return pre${return_new_axis}

    def post(self, gx):
        """
        Post-expression method.
        """
        % for param in coupling.parameters:
        ${param} = self.${param}
        % endfor

        post = ${post_expr}
        return post

    def __str__(self):
        return simple_gen_astr(self, "${" ".join(list(p.name for p in coupling.parameters.values()))}")
