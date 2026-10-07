# -*- coding: utf-8 -*-
<%
from tvbo.codegen import render_expression

eq, params = stimulus.get_expression()
stimulus_ident = stimulus.identifier
%>

%if jax:
import jax.numpy as jnp
import jax.scipy as jsp
%else:
import numpy as np
%endif


def ${stimulus_ident}(t, ${', '.join([f"{p}={v}" for p,v in params.items()])}):

    eq_t =  ${render_expression(eq, format='jax' if jax else 'numpy')}

    return eq_t

