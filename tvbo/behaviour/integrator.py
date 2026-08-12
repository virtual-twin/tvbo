"""Ontology-backed metadata, code generation and factories for :class:`Integrator`.

Attached to the generated classes by name (``IntegratorBehaviour`` -> ``Integrator``).
Only schema fields are ever stored; everything derived from the ontology is a property,
so an integrator can be serialized at any point without carrying runtime state.

Population from the ontology is NOT done at construction. It is
:meth:`IntegratorBehaviour.enrich`, called explicitly and idempotent, so that resolving an
integrator against the ontology stays an act the caller asks for rather than the cost of
naming one. The wrapper this replaces called it from ``__init__``, which meant it ran for
a directly constructed integrator and not for a loaded one; the loaded path compensated
with its own call, and that call is now the only one.

A mixin *can* hook construction — see :mod:`tvbo.behaviour` on ``__post_init__``, which
:class:`CouplingBehaviour` uses. This one deliberately does not.
"""

from __future__ import annotations


class IntegratorBehaviour:
    """Ontology metadata, rendering and loading for an integrator."""

    @classmethod
    def from_file(cls, filepath: str) -> "IntegratorBehaviour":
        """Load an Integrator from a YAML file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(str(filepath), target_class=cls)

    @classmethod
    def from_db(cls, name: str) -> "IntegratorBehaviour":
        """Load an Integrator by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Integrator", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available integrators in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("Integrator")

    @property
    def metadata(self):
        """The integrator itself, exposed for backward compatibility."""
        return self

    @property
    def ontoclass(self):
        """The ontology class for this integrator, resolved from `method`.

        Resolves a string `method` via the ontology, passes through an existing ontology
        `ThingClass`, and yields `None` otherwise.
        """
        import owlready2

        from tvbo.ontology import owl as ontology

        return (
            ontology.get_integrator(self.method)
            if isinstance(getattr(self, "method", None), str)
            else (self.method if isinstance(self.method, owlready2.entity.ThingClass) else None)
        )

    @property
    def info(self):
        """The code-generation metadata dict for this integrator's ontology class."""
        from tvbo.codegen import templater

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

        A plain datamodel `Noise` is upgraded to the runtime `Noise` subclass so it gains
        the computed properties and code-generation helpers.
        """
        from tvbo.classes.noise import Noise
        from tvbo.datamodel import schema as tvbo_datamodel

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
        return 0

    def enrich(self, source: str | None = None):
        """Fill the ontology-derived fields that are still unset. Idempotent.

        The same verb as :meth:`IriEnrichable.enrich`, and the same gap-filling contract,
        but its own implementation and no *key*: an integrator names an ontology method
        rather than a curated entity, so there is nothing for the database to answer and
        nothing to redirect the lookup at.
        """
        from tvbo.datamodel.schema import DerivedVariable, Equation
        from tvbo.ontology.owl import onto

        if source not in (None, "ontology"):
            raise ValueError(f"Integrator has no {source!r} to enrich from.")

        oc = self.ontoclass
        if not oc:
            return self
        info = self.info
        try:
            if hasattr(self, "scipy_ode_base"):
                self.scipy_ode_base = onto.SciPyODEBase in oc.is_a
        except Exception:
            pass

        if getattr(self, "intermediate_expressions", None) in (None, {}):
            steps = info.get("intermediate_steps", [])
            if steps:
                for i, eq in enumerate(steps):
                    self.intermediate_expressions[f"X{i + 1}"] = DerivedVariable(
                        name=f"X{i + 1}", equation=Equation(lhs=f"X{i + 1}", rhs=eq)
                    )
        if getattr(self, "number_of_stages", None) in (None, 0) and "n_dx" in info:
            self.number_of_stages = info["n_dx"]
        else:
            self.number_of_stages = len(self.intermediate_expressions) + 1

        if getattr(self, "update_expression", None) is None and "dX_expr" in info:
            self.update_expression = DerivedVariable(
                name="dX", equation=Equation(lhs="X_{t+1}", rhs=info["dX_expr"])
            )
        return self

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
        from tvbo import templates

        if format == "tvb":
            template = templates.lookup.get_template("tvbo-tvb-integration.py.mako")
            return template.render(integrator=self)
        if format.lower() in ["autodiff", "jax"]:
            template = templates.lookup.get_template("tvbo-jax-integrate.py.mako")
            return template.render(integration=self, **kwargs)
        raise ValueError(f"Unknown format: {format}")

    def execute(self, format="tvb"):
        """Render, execute, and instantiate the integrator backend object.

        For the `tvb` backend the integrator class is instantiated, wiring in an executed
        noise object when stochastic; for other backends the generated class is returned.

        Args:
            format:
                Target backend passed through to code generation.

        Returns:
            The executed backend integrator object or class.
        """
        from tvbo.codegen import templater

        local_vars = {}
        exec(self.render_code(format=format), templater.exec_globals, local_vars)

        if format.lower() != "tvb":
            return local_vars[self.class_name]
        params = {}
        if self.stochastic and self.noise_wrapper is not None:
            params["noise"] = self.noise_wrapper.execute()
        return local_vars[self.class_name](**params)

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
