# -*- coding: utf-8 -*-
<%
from tvbo.codegen import render_expression
from tvbo.templates.base.utils import safe_name

eq, params = stimulus.get_expression()
# The emitted function name has to be a Python identifier; `label` is free text.
stimulus_ident = safe_name(getattr(stimulus, 'label', None), fallback='stimulus')
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

