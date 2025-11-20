# -*- coding: utf-8 -*-
<%
from tvbo.export.code import render_expression
eq, params = stimulus.get_expression()
%>

%if jax:
import jax.numpy as jnp
import jax.scipy as jsp
%else:
import numpy as np
%endif


def ${stimulus.label }(t, ${', '.join([f"{p}={v}" for p,v in params.items()])}):

    eq_t =  ${render_expression(eq, format='jax' if jax else 'python')}

    return eq_t

