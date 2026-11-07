"""The module scope a rendered streaming reducer runs in.

A reducer is emitted as one ``<%def>`` and exercised here on its own, so the module-level helpers it calls -- the measurement-anchored output grid, the ring a kernel starts warm -- are not in the namespace unless something puts them there. Rendering them from the same template the reducer came from is what keeps a harness from freezing a stale copy: a helper whose contract changes changes here too, in the same commit, or the tests that depend on it stop running rather than quietly testing yesterday's rule.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mako.template import Template

OBS_TEMPLATE = Template(filename="tvbo/templates/tvboptim/tvbo-tvboptim-observation.py.mako")
"""The observation template, so a harness renders one `<%def>` out of the same file the backend emits from."""


def reducer_namespace(**extra) -> dict:
    """A namespace holding the reducer's module-level dependencies, plus whatever the caller adds.

    Args:
        extra: Names the reducer under test needs beyond the shared helpers -- a kernel
            function, a loaded constant -- bound as the emitted module would bind them.
    """
    namespace = {"jnp": jnp, "jax": jax}
    exec(compile(OBS_TEMPLATE.get_def("render_clock_helpers").render(), "<clock_helpers>", "exec"), namespace)
    namespace.update(extra)
    return namespace
