from tvb.datatypes.equations import *
import numpy
from tvb.basic.neotraits.api import Attr, Final

class ${eq_name}(${eq_type}):
    """
    This is a custom Equation class generated from a template.
    ${definition}
    """
    equation=Final(
        label="${eq_name}",
        default="${code_str}",
        doc="""${latex_str}"""
    )

    parameters=Attr(
        field_type=dict,
        label="Parameters for ${eq_name}",
        default=lambda: ${parameters}
    )