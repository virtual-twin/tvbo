"""Runtime `Noise` and `Integrator` wrappers around the TVBO datamodel classes.

These subclasses add computed properties (sigma/nsig, ontology-derived integrator metadata), JAX pytree registration, and code-generation/execution helpers on top of the plain serializable datamodel definitions, without introducing runtime caches or mutating stored parameters.
"""

import functools

import numpy as np
import owlready2
import sympy as sp
from jax.tree_util import register_pytree_node_class

from tvbo import templates
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.datamodel.schema import DerivedVariable, Equation
from tvbo.codegen import templater
from tvbo.ontology import owl as ontology
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
        if hasattr(self, "sigma_vec") and getattr(self, "sigma_vec") is not None:
            children = (getattr(self, "sigma_vec"),)
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
            setattr(obj, "sigma_vec", children[0])
        return obj

    @property
    def parameters_dict(self):
        """The noise parameters normalized to a dict-like view (empty dict if unset)."""
        # Normalize parameters to a dict-like view
        params = getattr(self, "parameters", None)
        return params if isinstance(params, dict) else (params or {})

    @property
    def symbolic(self):
        """The symbolic noise term $\\sqrt{dt}\\,\\sigma\\,\\xi$ for gaussian/white noise.

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
        """The noise dispersion `nsig`, derived from `sigma` as $0.5\\,\\sigma^2$ if needed.

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
        """The noise standard deviation `sigma`, derived from `nsig` as $\\sqrt{2\\,nsig}$ if needed.

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


class Integrator(tvbo_datamodel.Integrator):
    """Direct datamodel Integrator with ontology-backed population.

    Only schema fields are stored. All runtime conveniences are exposed as properties.
    """

    def __init__(self, **kwargs):
        # Accept either datamodel Noise, our subclass Noise, or a raw dict
        init_kwargs = dict(kwargs)
        n = init_kwargs.get("noise")
        if isinstance(n, dict):
            init_kwargs["noise"] = tvbo_datamodel.Noise(**n)
        # If it's already an instance of Noise or tvbo_datamodel.Noise, pass through
        super().__init__(**init_kwargs)

        self._populate_from_ontology()

    @classmethod
    def from_file(cls, filepath: str) -> "Integrator":
        """Load an Integrator from a YAML file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(str(filepath), target_class=cls)

    @classmethod
    def from_db(cls, name: str) -> "Integrator":
        """Load an Integrator by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Integrator", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available integrators in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("Integrator")

    # Back-compat: expose  pointing to self
    @property
    def metadata(self):
        """The integrator itself, exposed for backward compatibility."""
        return self

    # Runtime properties (no stored attributes)
    @property
    def ontoclass(self):
        """The ontology class for this integrator, resolved from `method`.

        Resolves a string `method` via the ontology, passes through an existing ontology `ThingClass`, and yields `None` otherwise.
        """
        return (
            ontology.get_integrator(self.method)
            if isinstance(getattr(self, "method", None), str)
            else (self.method if isinstance(self.method, owlready2.entity.ThingClass) else None)
        )

    @property
    def info(self):
        """The code-generation metadata dict for this integrator's ontology class."""
        return templater.get_integrator_info(self.ontoclass)

    @property
    def class_name(self):
        """The generated integrator class name, suffixed with `Stochastic` when noisy."""
        base = self.info.get("class_name", "Integrator")
        return base + ("Stochastic" if self.stochastic else "")

    @property
    def stochastic(self):
        """Whether the integrator is stochastic, i.e. has a noise component."""
        return bool(getattr(self, "noise", None))

    @property
    def noise_wrapper(self):
        """The noise as a runtime `Noise` wrapper, or `None` when non-stochastic.

        A plain datamodel `Noise` is upgraded to the runtime `Noise` subclass so it gains the computed properties and code-generation helpers.
        """
        if not self.stochastic:
            return None
        n = getattr(self, "noise", None)
        if isinstance(n, Noise):
            return n
        if isinstance(n, tvbo_datamodel.Noise):
            if hasattr(n, "_as_dict"):
                data = n._as_dict if not callable(n._as_dict) else n._as_dict()
                return Noise(**data)
            return Noise(**getattr(n, "__dict__", {}))
        return None

    @property
    def current_step(self):
        """The current integration step, a stateless default of `0`."""
        # Stateless default; templates can use this without mutating state
        return 0

    def _populate_from_ontology(self):
        oc = self.ontoclass
        if not oc:
            return
        info = self.info
        # scipy_ode_base (if present in schema)
        try:
            if hasattr(self, "scipy_ode_base"):
                self.scipy_ode_base = onto.SciPyODEBase in oc.is_a
        except Exception:
            pass

        # intermediate_expressions
        if getattr(self, "intermediate_expressions", None) in (None, {}):
            steps = info.get("intermediate_steps", [])
            if steps:
                for i, eq in enumerate(steps):
                    self.intermediate_expressions[f"X{i + 1}"] = DerivedVariable(
                        name=f"X{i + 1}", equation=Equation(lhs=f"X{i + 1}", rhs=eq)
                    )
        # number_of_stages
        if getattr(self, "number_of_stages", None) in (None, 0) and "n_dx" in info:
            self.number_of_stages = info["n_dx"]
        else:
            self.number_of_stages = len(self.intermediate_expressions) + 1

        # update_expression
        if getattr(self, "update_expression", None) is None and "dX_expr" in info:
            self.update_expression = DerivedVariable(name="dX", equation=Equation(lhs="X_{t+1}", rhs=info["dX_expr"]))

    def render_code(self, format="tvb", **kwargs):
        """Render the integrator as source code for the requested backend.

        Args:
            format:
                Target backend; `"tvb"` selects the TVB template, while `"autodiff"` or
                `"jax"` selects the JAX template.
            **kwargs:
                Extra values forwarded to the JAX template render context.

        Returns:
            The rendered source code as a string.

        Raises:
            ValueError: If `format` is not a recognized backend.
        """
        if format == "tvb":
            self.template = templates.lookup.get_template("tvbo-tvb-integration.py.mako")
            rendered_code = self.template.render(integrator=self)
        elif format.lower() in ["autodiff", "jax"]:
            self.template = templates.lookup.get_template("tvbo-jax-integrate.py.mako")
            rendered_code = self.template.render(integration=self, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")
        return rendered_code

    def execute(self, format="tvb"):
        """Render, execute, and instantiate the integrator backend object.

        For the `tvb` backend the integrator class is instantiated (wiring in an executed noise object when stochastic) and stored on `self.tvb`; for other backends the generated class is returned directly.

        Args:
            format:
                Target backend passed through to code generation.

        Returns:
            The executed backend integrator object or class.
        """
        local_vars = {}
        exec(self.render_code(format=format), templater.exec_globals, local_vars)

        if format.lower() == "tvb":
            params = {}
            if self.stochastic and self.noise_wrapper is not None:
                params.update({"noise": self.noise_wrapper.execute()})
            self.tvb = local_vars[self.class_name](**params)
            return self.tvb
        else:
            return local_vars[self.class_name]

    def to_yaml(self, filepath: str | None = None):
        """Serialize the integrator to YAML, optionally writing it to a file.

        Args:
            filepath:
                Destination path to write to; when `None`, the YAML is returned instead.

        Returns:
            The YAML string, or the result of writing to `filepath`.
        """
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, filepath)
