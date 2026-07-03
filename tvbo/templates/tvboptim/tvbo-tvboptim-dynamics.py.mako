# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Standalone Dynamics Template
=======================================

Renders a complete AbstractDynamics subclass with imports.
Used by dynamics.render_code('tvboptim').

Context Variables:
- model: Dynamics instance
</%doc>
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Tuple

from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics


<%include file="/tvboptim/tvbo-tvboptim-dfun.py.mako" />
