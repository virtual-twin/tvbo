<%
if 'experiment' in context.keys():
    stimulus = context['experiment'].metadata.stimulation
else:
    stimulus = context['stimulus'].metadata

from tvbo.knowledge.simulation.equations import (
    conditionals2piecewise,
    piecewise2numpy,
    _clash1,
    convert_ifelse_to_np_where,
)
from sympy import pycode, parse_expr, Symbol


if stimulus.equation.pycode:
    default_expression = stimulus.equation.pycode
elif stimulus.equation.conditionals:
    default_expression = convert_ifelse_to_np_where(
        pycode(
            conditionals2piecewise(stimulus.equation).subs("t", Symbol("var")),
            fully_qualified_modules=False,
        )
    )
else:
    default_expression = pycode(
        parse_expr(stimulus.equation.rhs, _clash1), fully_qualified_modules=False
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
