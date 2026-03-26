# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Standalone Coupling Template
=======================================

Renders a complete coupling class with imports.
Used by Coupling.render_code('tvboptim').

Context Variables:
- coupling: Coupling instance
</%doc>
import jax
import jax.numpy as jnp

from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling import InstantaneousCoupling, DelayedCoupling


<%include file="tvbo-tvboptim-cfun.py.mako" />
