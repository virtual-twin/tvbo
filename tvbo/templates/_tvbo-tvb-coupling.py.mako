<%!
    import numpy as np
%>
# Coupling Function
<%
if sparse:
    base_class = 'SparseCoupling'
else:
    base_class = 'Coupling'
%>

class ${class_name}(${base_class}):
    """
    This is a custom Coupling class generated from a template.
    It allows for specific pre and post expression definitions.
    """

    % for param, param_props in parameters.items():
    ${param} = NArray(
        label="${getattr(param_props, 'label', param)}",
        default=np.array([${getattr(param_props, 'value', '0')},]),
        domain=Range(lo=${getattr(param_props.domain, 'lo', '0.0')},
                     hi=${getattr(param_props.domain, 'hi', '1.0')},
                     step=${getattr(param_props.domain, 'step', '0.01')}),
        doc="${getattr(param_props, 'description', '')}"
    )
    % endfor

    parameter_names = ${list(parameters)}
    pre_expr = "${pre_expr.replace('[', '[:, ') if '[0]' in pre_expr else pre_expr}"
    post_expr = "${post_expr}"

<%
import re
def replace_params(expr, parameters):
    for p in sorted(parameters.values(), key=lambda x: len(x.name), reverse=True):
        expr = re.sub(r'\b{}\b'.format(re.escape(p.name)), 'self.' + p.name, expr)
    return expr

final_pre_expr = replace_params(pre_expr, parameters)
if '[0]' in final_pre_expr:
    final_pre_expr = final_pre_expr.replace('[', '[:, ')
    return_new_axis = "[:, np.newaxis]"
else:
    return_new_axis = ""
%>

    def pre(self, x_i, x_j):
        """
        Pre-expression method.
        """
        % for param in parameters:
        ${param} = self.${param}
        % endfor

        pre = ${final_pre_expr}
        return pre${return_new_axis}

    def post(self, gx):
        """
        Post-expression method.
        """
        % for param in parameters:
        ${param} = self.${param}
        % endfor

        post = ${replace_params(post_expr, parameters)}
        ## print(post)
        return post

    def __str__(self):
        return simple_gen_astr(self, "${" ".join(list(p.name for p in parameters.values()))}")
