import numpy as np
import owlready2
import sympy as sp
from jax.tree_util import register_pytree_node_class

from tvbo import templates
from tvbo.datamodel import tvbo_datamodel
from tvbo.datamodel.tvbo_datamodel import DerivedVariable, Equation
from tvbo.export import templater
from tvbo.knowledge import ontology
from tvbo.knowledge.ontology import onto

available_integrators = onto.IntegrationMethod.descendants(include_self=False)


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
        kwargs = (
            aux_data[0] if (isinstance(aux_data, tuple) and len(aux_data) > 0) else {}
        )
        if not isinstance(kwargs, dict):
            kwargs = {}
        obj = cls(**kwargs)
        # Reattach sigma_vec child if it was provided
        if isinstance(children, tuple) and len(children) == 1:
            setattr(obj, "sigma_vec", children[0])
        return obj

    @property
    def parameters_dict(self):
        # Normalize parameters to a dict-like view
        params = getattr(self, "parameters", None)
        return params if isinstance(params, dict) else (params or {})

    @property
    def symbolic(self):
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
        p = self.parameters_dict
        if "nsig" in p and p["nsig"] is not None:
            v = p["nsig"]
            return (
                getattr(v, "value", None) if not isinstance(v, dict) else v.get("value")
            )
        if "sigma" in p and p["sigma"] is not None:
            s = p["sigma"]
            s_val = (
                getattr(s, "value", None) if not isinstance(s, dict) else s.get("value")
            )
            if s_val is not None:
                return 0.5 * (s_val**2)
        return None

    @property
    def sigma(self):
        p = self.parameters_dict
        if "sigma" in p and p["sigma"] is not None:
            s = p["sigma"]
            return (
                getattr(s, "value", None) if not isinstance(s, dict) else s.get("value")
            )
        if "nsig" in p and p["nsig"] is not None:
            n = p["nsig"]
            n_val = (
                getattr(n, "value", None) if not isinstance(n, dict) else n.get("value")
            )
            if n_val is not None:
                return np.sqrt(2 * n_val)
        return None

    def render_code(self, format="tvb"):
        if format == "tvb":
            template = templates.lookup.get_template("tvbo-tvb-noise.py.mako")

        elif format.lower() in ["autodiff", "jax"]:
            template = templates.lookup.get_template("jax-noise.py.mako")
        rendered_code = template.render(
            noise=self,
        )
        return rendered_code

    def execute(self, format="tvb"):
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

    # Back-compat: expose .metadata pointing to self
    @property
    def metadata(self):
        return self

    # Runtime properties (no stored attributes)
    @property
    def ontoclass(self):
        return (
            ontology.get_integrator(self.method)
            if isinstance(getattr(self, "method", None), str)
            else (
                self.method
                if isinstance(self.method, owlready2.entity.ThingClass)
                else None
            )
        )

    @property
    def info(self):
        return templater.get_integrator_info(self.ontoclass)

    @property
    def class_name(self):
        base = self.info.get("class_name", "Integrator")
        return base + ("Stochastic" if self.stochastic else "")

    @property
    def stochastic(self):
        return bool(getattr(self, "noise", None))

    @property
    def noise_wrapper(self):
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
                    self.intermediate_expressions[f"X{i+1}"] = DerivedVariable(
                        name=f"X{i+1}", equation=Equation(lhs=f"X{i+1}", rhs=eq)
                    )
        # number_of_stages
        if getattr(self, "number_of_stages", None) in (None, 0) and "n_dx" in info:
            self.number_of_stages = info["n_dx"]
        else:
            self.number_of_stages = len(self.intermediate_expressions) + 1

        # update_expression
        if getattr(self, "update_expression", None) is None and "dX_expr" in info:
            self.update_expression = DerivedVariable(
                name="dX", equation=Equation(lhs="X_{t+1}", rhs=info["dX_expr"])
            )

    def render_code(self, format="tvb", **kwargs):
        if format == "tvb":
            self.template = templates.lookup.get_template(
                "tvbo-tvb-integration.py.mako"
            )
            rendered_code = self.template.render(integrator=self)
        elif format.lower() in ["autodiff", "jax"]:
            self.template = templates.lookup.get_template("tvbo-jax-integrate.py.mako")
            rendered_code = self.template.render(integration=self, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")
        return rendered_code

    def execute(self, format="tvb"):
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
        from tvbo.utils import to_yaml as _to_yaml

        return _to_yaml(self, filepath)
