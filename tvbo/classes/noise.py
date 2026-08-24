"""Runtime `Noise` wrapper, and the public import location for `Integrator`.

`Noise` is still a subclass: it registers itself as a JAX pytree, which is a decorator on the class being defined and so cannot come from a mixin. `Integrator` has no wrapper — its behaviour lives in :mod:`tvbo.behaviour.integrator`, attached to the generated class, and is re-exported here so the import path is unchanged.
"""

import functools

import numpy as np
import sympy as sp
from jax.tree_util import register_pytree_node_class

from tvbo import templates
from tvbo.codegen import templater
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.datamodel.schema import Equation
from tvbo.ontology.owl import onto


@functools.cache
def _available_integrators():
    """The ontology's integration-method classes, resolved and memoised on first use."""
    return onto.IntegrationMethod.descendants(include_self=False)


def __getattr__(name):  # PEP 562: keep ``available_integrators`` importable, lazily.
    # Resolving eagerly at import would force the ontology to load; deferred to first access.
    if name == "available_integrators":
        return _available_integrators()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@register_pytree_node_class
class Noise(tvbo_datamodel.Noise):
    """Datamodel Noise with property-based conveniences and no runtime caches.

    - Inherits all serializable fields from tvbo_datamodel.Noise directly.
    - Provides computed properties for sigma and nsig based on parameters.
    - No mutation of parameters and no cached fields; safe for serialization.
    """

    def __init__(self, **kwargs):
        if not self.equation:
            if self.noise_type in ("gaussian", "white"):
                self.equation = Equation(lhs="N", rhs="sqrt(dt) * sigma * xi")
            elif self.noise_type in ("ou", "ornstein-uhlenbeck"):
                self.equation = Equation(lhs="dN/dt", rhs="-N/tau + sigma * xi")
        super().__init__(**kwargs)

    # JAX pytree: carry no array children; aux holds serializable kwargs
    def tree_flatten(self):
        """Flatten into JAX pytree (children, aux).

        A present `sigma_vec` is exposed as the single array child so it can participate in `vmap` batching; the reconstruction kwargs go in aux.
        """
        aux = getattr(self, "_as_dict", None)
        if callable(aux):
            aux = aux()
        if aux is None:
            aux = dict(getattr(self, "__dict__", {}))
        # Do not include transient runtime fields in aux
        aux.pop("sigma_vec", None)
        # Expose sigma_vec (if present) as a child so it can participate in vmap batching
        children = ()
        if hasattr(self, "sigma_vec") and self.sigma_vec is not None:
            children = (self.sigma_vec,)
        return children, (aux,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a `Noise` instance from JAX pytree aux_data and children."""
        kwargs = aux_data[0] if (isinstance(aux_data, tuple) and len(aux_data) > 0) else {}
        if not isinstance(kwargs, dict):
            kwargs = {}
        obj = cls(**kwargs)
        # Reattach sigma_vec child if it was provided
        if isinstance(children, tuple) and len(children) == 1:
            obj.sigma_vec = children[0]
        return obj

    @property
    def parameters_dict(self):
        """The noise parameters normalized to a dict-like view (empty dict if unset)."""
        # Normalize parameters to a dict-like view
        params = getattr(self, "parameters", None)
        return params if isinstance(params, dict) else (params or {})

    @property
    def symbolic(self):
        r"""The symbolic noise term $\\sqrt{dt}\\,\\sigma\\,\\xi$ for gaussian/white noise.

        Returns `None` for noise types other than `gaussian`/`white`.
        """
        dt = sp.symbols("dt", real=True, positive=True)
        sigma_sym = sp.symbols("sigma", real=True, positive=True)
        xi = sp.symbols("xi", real=True)
        if isinstance(self.noise_type, str) and self.noise_type.lower() in (
            "gaussian",
            "white",
        ):
            return sp.sqrt(dt) * sigma_sym * xi

    @property
    def nsig(self):
        r"""The noise dispersion `nsig`, derived from `sigma` as $0.5\\,\\sigma^2$ if needed.

        Prefers an explicit `nsig` parameter; otherwise computes it from `sigma`. Returns `None` when neither is available.
        """
        p = self.parameters_dict
        if "nsig" in p and p["nsig"] is not None:
            v = p["nsig"]
            return getattr(v, "value", None) if not isinstance(v, dict) else v.get("value")
        if "sigma" in p and p["sigma"] is not None:
            s = p["sigma"]
            s_val = getattr(s, "value", None) if not isinstance(s, dict) else s.get("value")
            if s_val is not None:
                return 0.5 * (s_val**2)
        return None

    @property
    def sigma(self):
        r"""The noise standard deviation `sigma`, derived from `nsig` as $\\sqrt{2\\,nsig}$ if needed.

        Prefers an explicit `sigma` parameter; otherwise computes it from `nsig`. Returns `None` when neither is available.
        """
        p = self.parameters_dict
        if "sigma" in p and p["sigma"] is not None:
            s = p["sigma"]
            return getattr(s, "value", None) if not isinstance(s, dict) else s.get("value")
        if "nsig" in p and p["nsig"] is not None:
            n = p["nsig"]
            n_val = getattr(n, "value", None) if not isinstance(n, dict) else n.get("value")
            if n_val is not None:
                return np.sqrt(2 * n_val)
        return None

    def render_code(self, format="tvb"):
        """Render the noise as source code for the requested backend.

        Args:
            format:
                Target backend; `"tvb"` selects the TVB template, while `"autodiff"` or
                `"jax"` selects the JAX template.

        Returns:
            The rendered source code as a string.
        """
        if format == "tvb":
            template = templates.lookup.get_template("tvbo-tvb-noise.py.mako")

        elif format.lower() in ["autodiff", "jax"]:
            template = templates.lookup.get_template("jax-noise.py.mako")
        rendered_code = template.render(
            noise=self,
        )
        return rendered_code

    def execute(self, format="tvb"):
        """Render, execute, and instantiate the noise backend object.

        The rendered code is executed to obtain the `Noise` class, which is stored on `self.tvb` and returned.

        Args:
            format:
                Target backend passed through to code generation.

        Returns:
            The executed backend noise object.
        """
        local_vars = {}
        exec(self.render_code(), templater.exec_globals, local_vars)
        self.tvb = local_vars["Noise"]
        return self.tvb


Integrator = tvbo_datamodel.Integrator
"""The generated class itself. Its behaviour lives in :mod:`tvbo.behaviour.integrator` and
is attached where the class is generated, so an integrator carries it however it was
built — there is no wrapper to route construction through."""
