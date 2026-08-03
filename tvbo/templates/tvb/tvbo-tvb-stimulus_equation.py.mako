<%
if 'experiment' in context.keys():
    stimulus = context['experiment'].stimulation
else:
    stimulus = context['stimulus']

from tvbo.classes.equation import convert_ifelse_to_np_where
from tvbo.parse.symbols import BUILTIN_SHADOW
from sympy import pycode, Symbol


if stimulus.equation.pycode:
    default_expression = stimulus.equation.pycode
elif stimulus.equation.conditionals:
    default_expression = convert_ifelse_to_np_where(
        pycode(
            BUILTIN_SHADOW.parse(stimulus.equation).subs("t", Symbol("var")),
            fully_qualified_modules=False,
        )
    )
else:
    default_expression = pycode(
        BUILTIN_SHADOW.parse(stimulus.equation.rhs), fully_qualified_modules=False
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
