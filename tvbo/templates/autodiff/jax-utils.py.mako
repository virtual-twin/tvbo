# -*- coding: utf-8 -*-

<%
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

%>
## Helper that converts a array into a string that can be read as array again
<%def name = "array_input(array)" filter="trim">
        <%
        import numpy as np
        %>
        ## ${np.set_printoptions(threshold=sys.maxsize)}
        jnp.array(${np.array2string(array, separator = ",")})
</%def>

## Helper that converts a derived variable of interest into an expression that indexes from trace
<%def name = "generate_derived_expression(var, svars)" filter="trim">
        <%
        # Replace variable names with trace-indexed expressions
        for svar in svars:
            var = var.replace(svar, f"trace[:,[{svars.index(svar)}], :]")
        # Use JAX namespace when numpy appears
        var = var.replace("np.", "jnp.")
        var = var.replace("numpy.", "jnp.")
        %>
        ${var}
</%def>

## Backward-compat: keep the old misspelled name as an alias
<%def name = "gernerate_derived_expresion(var, svars)" filter="trim">
        ${generate_derived_expression(var, svars)}
</%def>
