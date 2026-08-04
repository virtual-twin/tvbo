<%
if 'experiment' in context.keys():
    stimulus = context['experiment'].stimulation
else:
    stimulus = context['stimulus']

from sympy import Symbol
from tvbo.codegen.code import get_printer

# TVB binds the stimulus argument as `var`, whatever the metadata calls time.
expression, _ = stimulus.get_expression()
default_expression = stimulus.equation.pycode or get_printer("tvb").doprint(
    expression.subs(Symbol("t"), Symbol("var"))
)
%>
################################################################################
from tvb.datatypes.equations import Equation, TemporalApplicableEquation
from tvb.basic.neotraits.api import Attr, Final
from numpy import where

class ${stimulus.label +'Equation'}(TemporalApplicableEquation):
    """
    This is a custom Equation class generated from a template.
    ${stimulus.description}
    """
    equation=Final(
        label="${stimulus.label }",
        default="${default_expression}",
    )

    parameters=Attr(
        field_type=dict,
        label="Parameters for ${stimulus.label }",
        default=lambda: ${{p.name: p.value for p in stimulus.parameters.values()}}
    )
